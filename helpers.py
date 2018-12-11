# consulted towards data science website to learn how to correctly parse using pandas and numpy arrays
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_raw_message(raw_message):
    lines = raw_message.split('\n')
    email = {}
    message = ''
    #print raw_message
    keys_to_extract = ['from', 'to', 'subject']
    for line in lines:
        if ':' not in line:
            message += line.strip()
            email['body'] = message
        else:
            pairs = line.split(':')
            key = pairs[0].lower()
            subject_split = ''
            for i in range(1, len(pairs)):
                if i == len(pairs) - 1:
                    subject_split += pairs[i].strip()
                else:
                    subject_split += pairs[i].strip() + ':'
            val = pairs[1].strip()
            if key in keys_to_extract:
                if key == 'subject':
                    email[key] = subject_split
                else:
                    email[key] = val
    return email

def parse_into_emails(messages):
    emails = [parse_raw_message(str(message)) for message in messages]
    return {
        'body': map_to_list(emails, 'body'),
        'to': map_to_list(emails, 'to'),
        'from_': map_to_list(emails, 'from'),
        'subject': map_to_list(emails, 'subject')
    }

def map_to_list(emails, key):
    results = []
    for email in emails:
        if key not in email:
            results.append('')
        else:
            results.append(email[key])
    return results



