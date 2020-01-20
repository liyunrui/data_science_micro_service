import os
import sqlite3
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from utils import remove_delimiter,remove_separator,remove_empty,remove_two_spaces,remove_three_spaces,df_to_txt


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/Users/william.teo/Downloads/sqlite.db', type=str, help = "data path")
    parser.add_argument('--output_path', default='../data', type=str, help = "path to save train/test text data ")
    parser.add_argument('--table_name', default='data', type=str, help='table name')
    parser.add_argument('--train_test_ratio', default=0.2, type=float, help='set ratio between 0 and 1 for train/test split')
    opt = parser.parse_args()

    # loda data
    conn = sqlite3.connect(opt.data_path)
    df = pd.read_sql_query("SELECT * FROM {}".format(opt.table_name), conn).drop(columns = ["index"])
    #
    #df = df.sample(100)
    # text cleaning
    df["text"] = df["text"].apply(lambda x : remove_delimiter(x))
    df["text"] = df["text"].apply(lambda x : remove_separator(x))
    df["text"] = df["text"].apply(lambda x : remove_empty(x))
    df["text"] = df["text"].apply(lambda x : remove_two_spaces(x))
    df["text"] = df["text"].apply(lambda x : remove_three_spaces(x))

    # train/test split
    assert 0 <= opt.train_test_ratio < 1
    df_train, df_test = train_test_split(df, test_size = opt.train_test_ratio)
    # save train/test result
    df_to_txt(df_train, os.path.join(opt.output_path,"train.txt"))
    df_to_txt(df_train, os.path.join(opt.output_path,"test.txt"))

if __name__ == "__main__":
    main()