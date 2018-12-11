# took help from stack overflow to solve some errors regarding file not parsing correctly for non UTF-8 characters
import os
import sys
from helpers import *
reload(sys)
sys.setdefaultencoding('utf8')

from collections import defaultdict
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

walk_dir = '../maildir'
messages = []

print('walk_dir = ' + walk_dir)

print('walk_dir (absolute) = ' + os.path.abspath(walk_dir))
# recrursively check files for each folder and sub-directory
for root, subdirs, files in os.walk(walk_dir):
    print('--\nroot = ' + root)
    # append all the mails to our m1 txt file
    list_file_path = 'm1.txt'
    print('list_file_path = ' + list_file_path)
    # opening in a+ mode to append to the file
    with open(list_file_path, 'a+') as list_file:
        for subdir in subdirs:
            print('\t- subdirectory ' + subdir)

        for filename in files:
            file_path = os.path.join(root, filename)

            print('\t- file %s (full path: %s)' % (filename, file_path))

            with open(file_path, 'rb') as f:
                f_content = f.read()
                messages.append(f_content)
                list_file.write('"'+f_content.replace('\r', ''))
                list_file.write('"\n')

# convert all msgs in panda table format for easy access
email_df = pd.DataFrame(parse_into_emails(messages))

# Drop emails with empty body, to or from_ columns.
email_df.drop(email_df.query("body == '' | to == '' | from_ == '' | subject == ''").index, inplace=True)
email_df.to_csv('dataset.csv')
parsed_emails = pd.read_csv('dataset.csv')
subj_dict = defaultdict(list)
for index, row in parsed_emails.iterrows():
    print index
    subject = row['subject']
    if len(subject.split(":")) == 1 or subject.split(":",1)[0].lower() != 're':
        try:
            sub = str(row['body'].encode('utf-8').strip())
            if not subject in subj_dict:
                if sub not in subj_dict[subject]:
                    subj_dict[subject].append(sub)
            else:
                if sub not in subj_dict[subject]:
                    subj_dict[subject].insert(0,sub)
        except:
            print "error"
    else:
        try:
            sub = str(row['body'].encode('utf-8').strip())
            cur = subject.split(":",1)[1]
            if sub not in subj_dict[cur]:
                subj_dict[cur].append(sub)
        except:
            print "error"

print len(subj_dict)
json.dump(subj_dict, open("email_response.json", 'w'))
