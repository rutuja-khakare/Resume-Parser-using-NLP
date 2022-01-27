from os import sep, write
from pandas.core.indexes.base import Index
import spacy
import pickle
import random
import re
import streamlit as st
import pandas as pd
import pdfminer
import os 
from pdfminer.high_level import extract_text
import shutil
import docx2txt
import zipfile
from spacy.matcher import Matcher
import re
import numpy as np
from nltk.corpus import stopwords
import spacy
from pyresparser import ResumeParser
import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
import pickle
import joblib
import streamlit.components as stc
import base64 
import time


nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)


def main():
    st.set_page_config(layout="wide")
    st.title('RESUME PARSER')
    train_data = pickle.load(open('C:\\Users\\admin\\Documents\\excel\\projects\\resume parser\\train_data.pkl', 'rb'))
    uploaded_files = st.file_uploader("Choose a file", type=['zip','pdf','docx','txt'],accept_multiple_files=False)
    if uploaded_files:
        st.success('File Uploaded Successfully')
        #st.write(uploaded_files.type)
        if uploaded_files.type == 'application/pdf':
            text = extract_pdf(uploaded_files)
            del_after_use()
        elif uploaded_files.type == 'text/plain':
            text = extract_txt(uploaded_files)
            del_after_use()

        elif uploaded_files.type == 'application/x-zip-compressed':
            text = extract_zip(uploaded_files)
            st.dataframe(text)
        else:
            text = extract_docx(uploaded_files)
            del_after_use()
            st.write(text)

        # Extracting information from Resume
        
        extract_info(text, uploaded_files.name)
def del_after_use():
    # Delete files from the folder after use
    folder = 'C:\\Users\\admin\\Documents\\excel\\projects\\resume parser\\Resumes'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            continue
            
def extract_docx(file):
    raw_text = docx2txt.process(file)
    return raw_text

def extract_pdf(file):
    with open(os.path.join("C:\\Users\\admin\\Documents\\excel\\projects\\resume parser\\Resumes",file.name),"wb") as f:
        f.write(file.getbuffer())

    pdf_path = f'C:\\Users\\admin\\Documents\\excel\\projects\\resume parser\\Resumes/{file.name}'
    text = ''
    for page in extract_text_from_pdf(pdf_path):
        text += ' '+ page
    return text

def extract_text_from_pdf(pdf_path):
    
    with open(pdf_path, 'rb') as fh:
        # iterate over all pages of PDF document
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            # creating a resoure manager
            resource_manager = PDFResourceManager()
            
            # create a file handle
            fake_file_handle = io.StringIO()
            
            # creating a text converter object
            converter = TextConverter(
                                resource_manager, 
                                fake_file_handle, 
                                codec='utf-8', 
                                laparams=LAParams()
                        )

            # creating a page interpreter
            page_interpreter = PDFPageInterpreter(
                                resource_manager, 
                                converter
                            )

            # process current page
            page_interpreter.process_page(page)
            
            # extract text
            text = fake_file_handle.getvalue()
            yield text

            # close open handles
            converter.close()
            fake_file_handle.close()
            
def extract_txt(file):
    raw_text = str(file.read(),"utf-8")
    return raw_text
def extract_zip(file):
    dir = 'C:\\Users\\admin\\Documents\\excel\\projects\\resume parser\\Resumes'
    # zip_path = store_zip(file)
    with zipfile.ZipFile(file.name) as zip_ref:
        zip_ref.extractall(dir)
    
    folder = [f for f in os.listdir(dir)]
    mypath = f'{dir}/{folder[0]}'
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(join(mypath, f))]
    
    text = []
    for file in onlyfiles:
        if file.lower().endswith('.docx'):
            text.append(extract_docx(f'{mypath}/{file}'))

        elif file.lower().endswith('.pdf'):
            text.append(extract_text_from_pdf(f'{mypath}/{file}'))

    # Adding fake email ids, phone numbers and links
    f_id = 'abc@xyz.com'
    f_no = '+911234567890'
    f_linkedin = 'https://www.linkedin.com/fake'
    f_github = 'https://www.github.com/fake'
    f_str = f_id+' '+f_no+' '+ f_linkedin+ ' '+f_github

    new_text = []
    for i in text:
        new_text.append(i+' '+f_str)

    # df = pd.DataFrame(text, columns=['Resume Text'])
    # st.write(len(text))
    del_after_use()
    return new_text
def extract_name(resume_text):
    isCapital = False
    if isCapital:
        return re.findall(r"[A-Z][A-Z]+\s+[A-Z][A-Z]+", resume_text)[0]
    else:
        nlp_text = nlp(resume_text)
        
        # First name and Last name are always Proper Nouns
        pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
        
        matcher.add('NAME', [pattern], on_match=None)
        
        matches = matcher(nlp_text)
        
        for match_id, start, end in matches:
            span = nlp_text[start:end]
            return span.text
    # df = pd.read_csv('FinalDF.csv')
    # return list(df['Name'])
def extract_phone(text):
    y = re.findall(r'[+]91[0-9]+|[0-9]{10}|[+?][0-9]+-[0-9]+| [+?][0-9]+\s[0-9]+',text)
    return y


def extract_email(text):
    email = re.findall("[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", text)
    if email:
        try:
            return email[0].split()[0].strip(';')
        except IndexError:
            return 'None'


def extract_links(text):
    y = re.findall(r'(https://(www.)?[a-z]+.com(/in)?/[a-z0-9]+)',text)    # Pattern to extract links
    links = []
    for strings in y:
        for link in strings:
            if len(link) > 4:
                links.append(link)
    return links

STOPWORDS = set(stopwords.words('english'))
def extract_info(text, filename):
          st.write(type(text))
             
# Education Degrees
EDUCATION = [
            'BE','B.E.', 'B.E', 'BS', 'B.S', 'B.SC', 'B E', 'B. E.','B. E','B S','B. S','B. SC'
            'ME', 'M.E', 'M.E.', 'MS', 'M.S', 'B-TECH','M-TECH','M E', 'M. E', 'M. E.', 'M S', 'M. S',
            'BTECH', 'B.TECH', 'M.TECH', 'MTECH','B TECH', 'B. TECH', 'M. TECH', 'M TECH',
            'B. TECH','M. TECH','B TECH','M TECH',
            'SSC', 'HSC', 'CBSE', 'ICSE', 'X', 'XII',
            'Bachelor of Technology','Senior Secondary'
            'BBA','B.B.A.','BCA','B.C.A.','BA','B.A.',
            'B.COM','B.ED','L.L.B.','LLB','LLM','L.L.M.',
            'MBA','M.B.A.','MCA','M.C.A.','MS','M.S.','MD',
            'M.D.','NDA','N.D.A.','PHD','PGDM','P.G.D.M.'
        ]

def extract_education(resume_text):
    nlp_text = sent_tokenize(resume_text)
    edu = {}
    # Extract education degree
    for index, text in enumerate(nlp_text):
        for tex in text.split():
            # Replace all special symbols
            tex = re.sub(r'[?|$|.|!|,]', r'', tex)
            if tex.upper() in EDUCATION and tex not in STOPWORDS:
                edu[tex] = text + nlp_text[index]   # Storing degree in a dictionary

# Extract year
    education = []
    if edu:
        for key in edu.keys():
            year = re.search(re.compile(r'(((20|19)(\d{2})))'), edu[key])   # Search for Year of Passing
            if year:
                education.append([key, ''.join(year[0])])
            else:
                education.append(key)
        return education
    else:
        EDU_PATTERN = re.findall(r'Bachelors? \D+|Masters? \D+',resume_text)    # If no degree is found, try searching for Bachelors or Masters in xyz
        return EDU_PATTERN
def extract_university(df):
    uni_df = pd.read_csv('List of Universities.csv')
    uni_df = uni_df.iloc[:,1:]
    f_uni = []

    for k in range(len(df)):    # Loop to input Resumes one at a time
        university_name = []
        lines = sent_tokenize(df['Resume Text'][k])          # Tokenizing text into sentences
        for sentence in lines:                               # Loop to run through each sentence
                if re.search('university',sentence.lower()) or re.search('education',sentence.lower()) or re.search('qualifications',sentence.lower()): # Search for words like education in sentence
                # print(sentence)
                    if re.search(r'\s{3,5}',sentence):  # If sentence has more than 3 spaces between words then split them
                        sens = re.split(r'\s{3,5}', sentence)
                        for i in sens:
                            if re.search('university',i.lower()) or re.search('education',i.lower()) or re.search('qualifications',i.lower()):  # Search for keywords like education in splitted sentences
                                edu_sen = i
                                # print(i[:100])
                                for j in uni_df['Name of University']:  # Loop to run through each university name
                                    if re.search(j.lower(),edu_sen.lower()):
                                        university_name.append(j)   # If university is found, append it.
                                        # print(j)

                    else:                               # If there are no 3-5 spaces found, then run following
                        edu_sen = sentence
                        for j in uni_df['Name of University']:      # Loop to run through each university name
                            if re.search(j.lower(),edu_sen.lower()):
                                university_name.append(j)           # If university is found, append it.
                

                    if university_name:
                        continue
                    else:
                        # print(sentence)
                        university_name.append(sentence)    # If particular university name not found, append whole sentence with keywords like education.

        # Remove duplicates    
        for i in range(len(university_name)-1):
            if university_name[i] == university_name[i+1]:
                del university_name[i]
        
        # Finally, appending all found universities and moving onto next resume
        f_uni.append(university_name)

    return f_uni

def extract_skills(text, filename):
    if filename.lower().endswith('.zip'):
        folder = [f for f in os.listdir(dir)]
        mypath = f'{dir}/{folder[0]}'
        onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(join(mypath, f))]
        for file in onlyfiles:
            skills = ResumeParser(f'{mypath}/{file}').get_extracted_data()['skills']
        return skills
    else:
        skills = ResumeParser(text).get_extracted_data()['skills']
        return skills
        
def extract_experience(text):
    lines = sent_tokenize(text)             # Sentence Tokenization
    experience = []
    for sentence in lines:
        if re.search('experience',sentence.lower()):        # Search for 'experience' keyword in sentence
            sen_tokenized = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(sen_tokenized)            # POS Tagging
            entities = nltk.chunk.ne_chunk(tagged)          # Structuring data into tree
            for subtree in entities.subtrees():
                for leaf in subtree.leaves():
                    # print(leaf)
                    if leaf[1] == 'CD':                     # Search for numerical values in the sentence
                        experience.append(leaf[0])
                        
    exp = []
    for ele in experience:
        if len(ele) <= 3 or (len(ele) <= 4 and ele[-1] == '0' 
                                and ele not in ('2020','2010','2000')):       # Finding relevant numerical value for experience
            exp.append(ele)
    if exp:
        return exp[0]
    else:
        return 'None'


def csv_downloader(data):
	csvfile = data.to_csv()
	b64 = base64.b64encode(csvfile.encode()).decode()
	new_filename = "new_text_file_{}_.csv".format(timestr)
	st.markdown("#### Download File ###")
	href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here!!</a>'
	st.markdown(href,unsafe_allow_html=True)
def extract_info(text, filename):
    # st.write(type(text))
    if filename.lower().endswith('.zip'):
        df = pd.DataFrame(text, columns=['Resume Text'])       
        # names = extract_name('None')
        names = []
        for text in df['Resume Text']:
            names.append(extract_name(text))

        phones = []
        for text in df['Resume Text']:
            phones.append(extract_phone(text))

        emails = []
        for text in df['Resume Text']:
            emails.append(extract_email(text))

        links = []
        for text in df['Resume Text']:
            links.append(extract_links(text))

        education = []
        for text in df['Resume Text']:
            education.append(extract_education(text))
        
        skills = []
        for text in df['Resume Text']:
            skills.append(extract_education(text))

        f_uni = extract_university(df)
        uni_list = []
        for ele in f_uni:
            uni_list.append(clean_data(ele))

        experience = []
        for text in df['Resume Text']:
            experience.append(extract_experience(text))
        
        names = pd.Series(names)
        details_df = pd.DataFrame(names, columns= ['Name'])
        details_df['Phone No.'] = phones
        details_df['Email ID'] = emails
        details_df['Links'] = links
        details_df['Skills'] = skills
        details_df['Education degree'] = education
        details_df['University'] = uni_list
        details_df['Years of Experience'] = experience

        # details_df.to_pickle('Details_df.pkl')
        # with open('Details_df.pkl','rb') as f:
        #     details_df = pickle.load(f)
        # details_df = pd.read_csv('FinalDF.csv')
        # details_df = details_df.iloc[:,1:]
        st.dataframe(details_df)
        csv_downloader(details_df)

      

        # st.download_button(
        # label="Download data as CSV",
        # data=details_df,
        # file_name='large_df.csv',
        # mime='text/csv',
        # )
        # st.download_button(label="Download data as CSV", data = details_df, file_name='data.csv', mime='text/csv')

    else:
        series = pd.Series(text)
        df = pd.DataFrame(series, columns=['Resume Text'])
        
        name = extract_name(text)
        st.header('Name')
        st.write(name)
        phone = extract_phone(text)
        
        st.header('Contact')
        st.write(phone)
        email = extract_email(text)
        
        st.header('Email ID')
        st.write(email)
        links = extract_links(text)
        
        st.header('Useful Links')
        st.write(links)
        skills = extract_skills(text, filename)
        
        st.header('Skills')
        st.write(skills)
        education = extract_education(text)
        
        st.header('Education Details')
        st.write(education)
        university = extract_university(df)
        
        st.header('University Name')
        st.write(university)
        experience = extract_experience(text)
        
        st.header('Years of Experience')
        st.write(experience)



        series2 = pd.Series(name)
        details_df = pd.DataFrame(series2, columns= ['Name'])
        details_df['Phone No.'] = phone
        details_df['Email ID'] = email
        details_df['Links'] = str(links)
        details_df['Skills'] = skills
        details_df['Education degree'] = education
        details_df['University'] = university
        details_df['Years of Experience'] = experience
        st.dataframe(details_df)
        csv_downloader(details_df)

        


if __name__ == '__main__':
    main()