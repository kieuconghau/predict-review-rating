#!/usr/bin/env python
# coding: utf-8

# <h1 align='center'>Đồ án 4: Phân lớp văn bản với kĩ thuật bình phương tối tiểu</h1>
# 
# Họ và tên | MSSV | Lớp
# ----------|------|------
# Kiều Công Hậu | 18127259 | 18CLC1

# In[1]:


# DON'T CHANGE this part: import libraries
import numpy as np
import scipy
import json
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import re
import itertools


# In[2]:


# DON'T CHANGE this part: read data path
train_set_path, valid_set_path, random_number = input().split()


# ## a. Xử lý dữ liệu văn bản

# Đọc dữ liệu của 1 cột thuộc tính, trong đó:
# - `data_path`: (str) đường dẫn của tập dữ liệu
# - `attribute`: (str) thuộc tính mà ta muốn đọc (ví dụ: reviewText, overall,...)
# 
# Cụ thể, em đã dùng hàm `json.load()` để đọc file .json.

# In[3]:


# Đọc file dữ liệu ứng với thuộc tính tương ứng.
def read_data(data_path, attribute):
    f = open(data_path)
    data = json.load(f)
    attribute_data = np.array([data[i][attribute] for i in range(len(data))])
    f.close()
    return attribute_data


# Tiền xử lý dữ liệu của một văn bản, trong đó:
# - `text`: (str) nội dung của văn bản mà ta muốn xử lý.
# 
# Cụ thể, em đã xài các hàm sau để xử lý:
# - Chuyển tất cả thành chữ thường: `lower()`
# - Chuyển số thành ký tự `'num'` (int lẫn float): `isnumeric()` kết hợp với `replace()`
# - Tách từ: `word_tokenize()`
# - Loại bỏ stopwords: `stopwords.words('english')`
# - Stemming: `PorterStemmer().stem()`
# 
# Từ `'unk'` cũng được thêm vào vocab. Cho nên, tất cả các từ out-of-vocab đều được đổi thành `'unk'`.

# In[4]:


# Tiền xử lý dữ liệu một văn bản.
def preprocess(text):
    filtered_word_list = []
    word_list = word_tokenize(text.lower())
    stop_word_list = stopwords.words('english')
    ps = PorterStemmer()
    
    for word in word_list:
        if word.replace('.', '', 1).isnumeric():
            word = 'num'
        if word not in stop_word_list:
            filtered_word_list.append(ps.stem(word))
            
    return np.array(filtered_word_list)


# Hàm dưới đây chỉ đơn giản là tiền xử lý nhiều văn bản và trả về một list các văn bản đã được tiền xử lý, trong đó:
# - `text_data`: list các văn bản

# In[5]:


# Tiền xử lý dữ liệu nhiều văn bản.
def preprocess_data(text_data):
    return np.array([preprocess(reviewText) for reviewText in text_data])


# Xây dựng vocab dựa vào bộ dữ liệu reviewText đã qua tiền xử lý, trong đó:
# - `pre_reviewText_data`: list các văn bản review đã qua tiền xử lý.
# 
# Cụ thể, vocab có cấu trúc dữ liệu dạng `dict`, key là từ, value tần số xuất hiện của từ đó.

# In[6]:


# Xây dựng vocab dựa trên dữ liệu đã được tiền xử lý.
def generate_vocab(pre_reviewText_data):
    vocab = {}
    for pre_reviewText in pre_reviewText_data:
        for word in pre_reviewText:
            vocab[word] = 0
    vocab['unk'] = 0
    return vocab


# Xây dụng ma trận histogram dựa vào list các văn bản review đã qua xử lý và vocab, trong đó:
# - `pre_reviewText_data`: list các văn bản review đã qua tiền xử lý.
# - `vocab`: bộ từ vựng.
# 
# Ý tưởng, tạo các `count vector` trước, sau đó chuyển `count vector` thành các `histogram vector`.

# In[7]:


# Xây dựng bộ ma trận nhúng của văn bản.
def generate_histogram_matrix(pre_reviewText_data, vocab):
    count_matrix = [vocab.copy() for _ in range(pre_reviewText_data.shape[0])]
    for i in range(pre_reviewText_data.shape[0]):
        for word in pre_reviewText_data[i]:
            if word not in vocab:
                word = 'unk'
            count_matrix[i][word] += 1

    count_matrix = np.array([np.array([count_vector[word] for word in count_vector]) for count_vector in count_matrix])
    histogram_matrix = count_matrix / (count_matrix @ np.ones(count_matrix.shape[1]))[:, np.newaxis]
    
    return histogram_matrix


# Chuẩn hóa danh sách các nhãn thành danh sách các vector, trong đó:
# - `overall_data`: list các đánh giá dựa trên văn bản review tương ứng.
# 
# Cụ thể, kết quả trả về của hàm này là một ma trận có kích thước `(n, 5)` với `n` là số lượng văn bản của tập train và `5` là số loại đánh giá {1, 2, 3, 4, 5}. Giả sử, một vản bản có `overall` là 3 thì vector tương ứng sẽ là `(0, 0, 1, 0, 0)`. Tổng quát, nếu `overall` là `i` thì phần tử tại index `i - 1` sẽ được bật lên 1, còn lại là 0.

# In[8]:


# Chuẩn hóa nhãn thành vector.
def generate_label_matrix(overall_data):
    label_matrix = np.zeros((overall_data.shape[0], 5))
    for i in range(overall_data.shape[0]):
        label_matrix[i][int(overall_data[i]) - 1] = 1
    return label_matrix


# ## b. Sử dụng mô hình hồi quy tuyến tính dùng bình phương tối tiểu

# Xây dựng mô hình hồi quy tuyến tính, trong đó:
# - `pre_train_reviewText_data`: list các văn bản review đã qua tiền xử lý của tập train.
# - `vocab`: bộ từ vựng từ tập train.
# - `train_overall_data`: list các đánh giá tương ứng với văn bản của tập train.

# In[9]:


# Xây dựng mô hình hồi quy tuyến tính dựa trên tập huấn luyện.
def linear_regression(pre_train_reviewText_data, vocab, train_overall_data):
    histogram_matrix = generate_histogram_matrix(pre_train_reviewText_data, vocab)
    label_matrix = generate_label_matrix(train_overall_data)
    x_hat = np.linalg.pinv(np.concatenate((np.ones((histogram_matrix.shape[0], 1)), histogram_matrix), axis=1)) @ label_matrix
    return x_hat


# ## c. Sử dụng độ chính xác để đánh giá mô hình

# Dựa vào mô hình đã xây dựng (x_hat), ta dễ dàng dự đoán được nhãn của các văn bản review cần kiểm thử thông qua công thức `y = A @ x_hat`. Sau đó đếm xem có bao nhiêu nhãn dự đoán đúng thông qua mô hình trên. Độ chính xác của mô hình được đánh giá bằng công thức dưới đây:
# $$ acc = \frac{\sum(right)}{\sum(total)} $$

# In[10]:


# Đánh giá độ chính xác của mô hình.
def calc_accuracy(x_hat, vocab, valid_reviewText_data, valid_overall_data):
    pre_valid_reviewText_data = preprocess_data(valid_reviewText_data)
    valid_histogram_matrix = generate_histogram_matrix(pre_valid_reviewText_data, vocab)
    y = np.concatenate((np.ones((valid_histogram_matrix.shape[0], 1)), valid_histogram_matrix), axis=1) @ x_hat
    predicted_label_data = np.argmax(scipy.special.softmax(y, axis=1), axis=1) + 1
    valid_count = 0
    for i in range(valid_overall_data.shape[0]):
        if valid_overall_data[i] == predicted_label_data[i]:
            valid_count += 1
    return valid_count / valid_overall_data.shape[0]


# ## main

# **Báo cáo tổng kết:** với 2 tập dataset `train.json` và `valid.json` trong file mô tả yêu cầu thì độ chính xác của mô hình M2 là 0.52 (thời gian chạy khoảng 60s).

# In[11]:


# Read data.
train_reviewText_data = read_data(train_set_path, 'reviewText')
train_overall_data = read_data(train_set_path, 'overall')

valid_reviewText_data = read_data(valid_set_path, 'reviewText')
valid_overall_data = read_data(valid_set_path, 'overall')

# Preprocess.
pre_train_reviewText_data = preprocess_data(train_reviewText_data)
vocab = generate_vocab(pre_train_reviewText_data)

# Linear regression.
x_hat = linear_regression(pre_train_reviewText_data, vocab, train_overall_data)

# Accuracy.
accuracy = calc_accuracy(x_hat, vocab, valid_reviewText_data, valid_overall_data)

# Output
print(list(pre_train_reviewText_data[int(random_number)]))
print("M2 - ", end='')
print(accuracy)


# In[ ]:




