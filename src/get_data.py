import argparse
import base64
import pandas as pd

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', default='../data/Users/william.teo/Downloads/db.sqlite.base64', type=str, help = "data path to base64 representation of the SQLite database")
    parser.add_argument('--output_path', default='../data/Users/william.teo/Downloads/sqlite.db', type=str, help = "path to binary of the SQLite database")
    opt = parser.parse_args()

    # load data
    with open(opt.raw_data_path, 'rb') as f:
        data = f.read()
        data = base64.b64decode(data)
        with open(opt.output_path,"wb") as g:
            g.write(data)


if __name__ == "__main__":
    main()