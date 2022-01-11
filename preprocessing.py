import email
import os
import re
from urllib.parse import urlparse

from shutil import rmtree

import numpy as np

URL_REGEX = r'((http|ftp|https):\/\/([\w\-_]+(?:(?:\.[\w\-_]+)+))([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?)'

DOT_LIMIT = 3
MAX_NEGATIVE_EXAMPLES = 10 ** 5


class Mail:
    class Attachment:
        def __init__(self, attachment_type, payload):
            self.type = attachment_type
            self.payload = payload

    TEXT_TYPES = ["text/html", "text/plain"]

    @staticmethod
    def load(path):
        data = open(path, "rb").read()
        msg = email.message_from_bytes(data)
        sender = str(msg["From"])
        if "<" in sender:
            sender = sender.split("<")
            for part in sender:
                if "@" in part:
                    sender = part.split(">")[0]
        assert "@" in sender
        receiver = msg["Delivered-To"]
        subject = msg["Subject"]
        text_types = list(set([x.get_payload() for x in list(msg.walk()) if x.get_content_type() in Mail.TEXT_TYPES]))
        if len(text_types) == 0:
            text_type = None
        else:
            text_type = "text/html" if "text/html" in text_types else text_types[0]

        body = "\n\n".join([x.get_payload() for x in list(msg.walk()) if x.get_content_type() in Mail.TEXT_TYPES])
        attachments = [Mail.Attachment(x.get_content_type(), x.get_payload())
                       for x in list(msg.walk()) if x.get_content_type() not in Mail.TEXT_TYPES]

        return Mail(sender=sender, receiver=receiver, subject=subject, body=body, attachments=attachments,
                    text_type=text_type)

    def __init__(self, sender, receiver, subject, body, attachments, text_type):
        self.sender = sender
        self.receiver = receiver
        self.subject = subject
        self.body = body
        self.attachments = attachments
        self.text_type = text_type


def check_for_urls_with_ip(mail):
    ip_candidates = re.findall(r"((www\.|http://|https://)\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", mail.body)
    return [0 if len(ip_candidates) == 0 else 1]


def check_link_in_href(mail):
    links = re.findall(r"<a .*</a>", mail.body, re.IGNORECASE)
    for link in links:
        link = link.lower()
        if "href=" not in link:
            continue
        inside = ">".join(link.split("</a>")[0].split(">")[1:]).strip()
        if re.match(URL_REGEX, inside):
            href = link.split("href=")[1]
            href = re.findall(URL_REGEX, href)[0][0]

            if inside not in href and href not in inside:
                return [1]

    return [0]


def check_catchphrase_inside_link(mail):
    links = re.findall(r"<a .*</a>", mail.body, re.IGNORECASE)
    catchphrases = ["click", "link", "here", "update", "login"]
    for link in links:
        link = link.lower()
        if "href=" not in link:
            continue
        inside = ">".join(link.split("</a>")[0].split(">")[1:]).strip()
        for catchphrase in catchphrases:
            if catchphrase in inside:
                return [1]
    return [0]


def check_multiple_dots_inside_urls(mail):
    urls = re.findall(URL_REGEX, mail.body, re.IGNORECASE)
    for url in urls:
        if url.count(".") > DOT_LIMIT:
            return [1]
    return [0]


def check_sender_domain_match(mail):
    sender_domain = mail.sender.split("@")[1]
    urls = [x[0] for x in re.findall(URL_REGEX, mail.body)]
    result = 0
    for url in urls:
        domain = urlparse(url).netloc

        if sender_domain not in domain:
            result += 1

    return [result]


def check_text_type(mail):
    return [1] if mail.text_type == "text/html" else [0]


def check_javascript_presence(mail):
    return [1] if "<script>" in mail.body or "javascript" in mail.body else [0]


def check_verification_words_presence(mail):
    verification_words = ["verify", "account", "notify", "credentials"]
    count = 0
    for word in verification_words:
        count += mail.body.lower().count(word)
    return [count]


def check_login_words_presence(mail):
    verification_words = ["login", "username", "password",  "click",  "log"]
    count = 0
    for word in verification_words:
        count += mail.body.lower().count(word)
    return [count]


def check_confirm_words_presence(mail):
    verification_words = ["update", "confirm"]
    count = 0
    for word in verification_words:
        count += mail.body.lower().count(word)
    return [count]


def extract_features(mail, y_value):
    features = [1]  # interceptor
    features = features + check_for_urls_with_ip(mail)
    features = features + check_link_in_href(mail)
    features = features + check_catchphrase_inside_link(mail)
    features = features + check_multiple_dots_inside_urls(mail)
    features = features + check_sender_domain_match(mail)
    features = features + check_text_type(mail)
    features = features + check_javascript_presence(mail)
    features = features + check_verification_words_presence(mail)
    features = features + check_login_words_presence(mail)
    features = features + check_confirm_words_presence(mail)

    return features + [y_value]


def load_positive_examples(start_path="positive_examples"):
    X = []

    for path in os.walk(start_path):
        root, dirs, files = path
        for file in files:
            file = os.path.join(root, file)
            try:
                mail = Mail.load(file)
            except AssertionError:
                continue
            X.append(extract_features(mail, 1))
    X = np.array(X)

    return X


def load_negative_examples(start_path="negative_examples"):
    X = []
    for path in os.walk(start_path):
        root, dirs, files = path
        if len(files) == 0:
            continue

        if "inbox" not in root or "notes_inbox" in root:
            rmtree(root)

    for path in os.walk(start_path):
        root, dirs, files = path
        for file in files:
            file = os.path.join(root, file)
            try:
                mail = Mail.load(file)
            except AssertionError:
                continue
            X.append(extract_features(mail, 0))
    X = np.array(X)

    return X


def get_data():
    K = 2  # pomer negativnych k pozitivnym

    try:
        print("loading ... ", end="")
        data = np.load("all_examples.dat", allow_pickle=True)
        print("loaded")
        return data
    except Exception as ex:
        print(f"failed with exception {ex}")
        print("extracting dataset ...", end="")
        positive = load_positive_examples()
        negative = load_negative_examples()
        if len(positive) > len(negative):
            p = len(negative) * K
            if p < len(positive):
                np.random.shuffle(positive)
                positive = positive[:p, :]
            all_examples = np.append(positive, negative, axis=0)
        else:
            p = len(positive) * K
            if p < len(negative):
                np.random.shuffle(negative)
                negative = negative[:p, :]
            all_examples = np.append(positive, negative, axis=0)

        all_examples.dump("all_examples.dat")
        print("done")

        return all_examples

