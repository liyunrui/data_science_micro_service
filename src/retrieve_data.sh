tar -C ../data -zvxf ../data/db.sqlite.base64.tar.gz
python3 get_data.py
python3 data_process.py