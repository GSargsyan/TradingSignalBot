import os
import json
import time
import smtplib
import sqlite3
import requests
from zoneinfo import ZoneInfo
from email.mime.text import MIMEText
from datetime import datetime, timedelta

import feedparser
from google import genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


def get_last_news(count, source):
    if source == 'google':
        url = os.getenv("GOOGLE_NEWS_URL")
    elif source == 'nbc':
        url = os.getenv("NBC_URL")
    elif source == 'politico':
        url = os.getenv("POLITICO_URL")
    elif source == 'abc':
        url = os.getenv("ABC_URL")
    else:
        raise ValueError(f"Invalid source: {source}")

    feed = feedparser.parse(url)
    news = [
        {"title": entry.title.strip(),
         "timestamp": entry.published,
         "url": entry.link}
        for entry in feed.entries[:count]]

    with open("news_log.ndjson", "a") as f:
        for entry in news:
            f.write(json.dumps({
                "source": source,
                "news_time": entry["timestamp"],
                "analyzed_at": str(datetime.now()),
                "url": url,
                "news": entry["title"]
            }) + "\n")

    return news


def classify_news_type(title):
    prompt = (
        "You are an analyst helping a trading algorithm understand how urgent and impactful a news headline is. "
        "You are given a single news title. Classify it as one of the following: "
        "- 'breaking': If it's an immediate, sudden, impactful news event (e.g. attack, death, announcement with instant effect). "
        "- 'routine': If it's commentary, opinion, forecast, ongoing situation, or future speculation. "
        "Here are some examples and what they should be classified as: "
        "1. Israel launches surprise airstrikes on Tehran - breaking"
        "2. Trump will decide on U.S. involvement in Iran within two weeks, White House says - routine (because feels like it's a follow up of an ongoing situation)"
        "3. Oil prices jump as Middle East conflict escalates overnight - routine (because oil prices are already affected by the situation)"
        "4. Analysts warn of long-term energy market instability - routine"
        "5. Russia invades Ukraine: NATO calls emergency meeting - breaking"
        "6. Explosion hits major oil facility in Saudi Arabia - breaking"
        "7. Federal Reserve unexpectedly hikes interest rates by 0.5% - breaking"
        "Reply with just one word: breaking or routine. "
        "Here is the title: {title}"
    )

    api_key = os.getenv("GEMINI_KEY")
    gemini_client = genai.Client(api_key=api_key)

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=prompt
    )

    return response.text


def analyze_news():
    prompt_template = (
        "You are a day trader, you are given a news article title and short "
        "description. Your task is to decide whether this news introduces new, "
        "market-moving information that could influence the price of stocks, "
        "commodities, or crypto assets within the next 12-24 hours. Most of the "
        "news will not affect any and you should reply by simply 'stand', but if "
        "they are likely to increase or decrease a value you should reply with "
        "'buy (or sell) - [resource name] - [the code/symbol]'. Only react to "
        "big shocking news. This is the news:\n\n"
        "{news_title}"
        "{similar_news_template}"
    )

    similar_news_template = (
        "The following are recent news headlines and their timestamps that seem "
        "similar or related. They may indicate that the current event is part of "
        "an ongoing situation. If so reply with 'stand'.\n\n"
        "{similar_news}"
    )

    api_key = os.getenv("GEMINI_KEY")
    gemini_client = genai.Client(api_key=api_key)

    log_output('Getting news...')
    news = []
    news.extend(get_last_news(count=3, source='google'))
    news.extend(get_last_news(count=3, source='nbc'))
    news.extend(get_last_news(count=3, source='politico'))
    news.extend(get_last_news(count=3, source='abc'))

    log_output(f'Got {len(news)} news')

    log_output('Getting already processed news...')
    seen = get_processed_titles()

    log_output(f'Got {len(seen)} already processed news')

    for entry in news:
        if entry["title"] in seen:
            continue 

        log_output(f'Analyzing news: {entry["title"]}')
        log_output('Getting similar news...')

        similar_news = get_similar_news(entry["title"])

        log_output(f'Got {len(similar_news.split("\n"))} similar news')

        prompt = prompt_template.format(news_title=entry["title"], similar_news_template=similar_news_template.format(similar_news=similar_news))

        log_output('Generating response...')
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            contents=prompt
        )

        log_output(f'Response: {response.text}')

        log_output('Logging data...')
        log_data = {
            "timestamp": str(datetime.now()),
            "title": entry["title"],
            "cached_content_token_count": response.usage_metadata.cached_content_token_count,
            "prompt_token_count": response.usage_metadata.prompt_token_count,
            "thoughts_token_count": response.usage_metadata.thoughts_token_count,
            "total_token_count": response.usage_metadata.total_token_count,
            "response": response.text,
            "prompt": prompt
        }

        if "buy" in response.text.lower() or "sell" in response.text.lower():
            # send_email(title, response.text, "g.sargsyan1995@gmail.com", "thesignalproject7@gmail.com")
            message = f"{entry['title']}\n\n{entry['url']}\n\n{response.text}"
            send_telegram_message(message)

        save_to_db(log_data)

        log_output('Done!')


def get_processed_titles():
    conn = sqlite3.connect('signals.db')
    query = ("SELECT title FROM signals")
    cursor = conn.execute(query)

    processed_news = set(row[0] for row in cursor)
    conn.close()

    return processed_news


def save_to_db(log_data):
    conn = sqlite3.connect('signals.db')
    query = ("INSERT INTO signals ("
             "timestamp,"
             "title,"
             "cached_content_token_count,"
             "prompt_token_count,"
             "thoughts_token_count,"
             "total_token_count,"
             "response,"
             "prompt) VALUES (:timestamp, :title, :cached_content_token_count, :prompt_token_count, :thoughts_token_count, :total_token_count, :response, :prompt)")

    conn.execute(query, log_data)
    conn.commit()
    conn.close()


def send_email(subject, body, to_email, from_email):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, os.getenv("GMAIL_PASSWORD"))
            server.sendmail(from_email, to_email, msg.as_string())
            log_output("✅ Email sent.")
    except Exception as e:
        log_output("❌ Failed to send email:", e)


def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{os.getenv('TELEGRAM_BOT_TOKEN')}/sendMessage"
    data = {
        "chat_id": os.getenv('TELEGRAM_USER_ID'),
        "text": message
    }
    requests.post(url, json=data)


def get_latest_news():
    conn = sqlite3.connect('signals.db')
    query = ("SELECT timestamp, title FROM signals "
             "WHERE timestamp > datetime('now', '-72 hours')")
    cursor = conn.execute(query)
    latest_news = {row[0]: row[1] for row in cursor}
    conn.close()

    return latest_news


def get_similar_news(title):
    sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_current = sentence_transformer.encode(title)

    latest_news = get_latest_news()
    similar_news = {}

    for timestamp, prev_title in latest_news.items():
        embedding_prev = sentence_transformer.encode(prev_title)
        similarity = float(sentence_transformer.similarity(embedding_current, embedding_prev)[0])

        if similarity > 0.3:
            title_with_timestamp = f"{timestamp} - {prev_title}"
            similar_news[title_with_timestamp] = similarity

    similar_news = sorted(similar_news.items(), key=lambda x: x[1], reverse=True)

    return "\n".join([f"{title}" for title, _ in similar_news[:3]])


def test_reuters_rss():
    news_url = os.getenv("NBC_URL")
    feed = feedparser.parse(news_url)

    # print titles of top 5 entries 
    for entry in feed.entries[:5]:
        print(entry.title)


def test_classify_news_type():
    print(classify_news_type("Trump will decide on U.S. involvement in Iran within two weeks, White House says"))


def test_similarity():
    sentences = [
        "The US bunker-buster bomb and how it could be used against Iran's nuclear program - ABC News",
        "Trump urges immediate evacuation of Tehran - The Hill",
        "Trump team proposes Iran talks this week on nuclear deal, ceasefire - Axios",
        "Israel's air superiority lets it strike Iran on the cheap \u2014 and force Tehran into costly retaliation - Business Insider",
        "Suspect in Minnesota shootings visited other legislators\u2019 homes, say authorities - The Guardian",
    ]
    sentence = "Donald Trump says he \u2018may or may not\u2019 strike Iran - Financial Times"

    sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
    sentences_emb = sentence_transformer.encode(sentences)
    sentence_emb = sentence_transformer.encode(sentence)

    sim_scores = sentence_transformer.similarity(sentences_emb, sentence_emb)

    for score in sim_scores:
        print(float(score))


def setup_db():
    if not os.path.exists('signals.db'):
        conn = sqlite3.connect('signals.db')
        conn.execute("CREATE TABLE signals ("
                     "timestamp DATETIME,"
                     "title TEXT,"
                     "cached_content_token_count INTEGER,"
                     "prompt_token_count INTEGER,"
                     "thoughts_token_count INTEGER,"
                     "total_token_count INTEGER,"
                     "response TEXT,"
                     "prompt TEXT)")
        conn.execute('CREATE INDEX idx_timestamp ON signals (timestamp)')
        conn.close()


def log_output(message: str):
    with open("output.log", "a") as f:
        f.write(f"{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")


if __name__ == '__main__':
    start_time = time.time()

    load_dotenv()
    setup_db()
    analyze_news()

    log_output(f"Total execution time: {time.time() - start_time} seconds\n")