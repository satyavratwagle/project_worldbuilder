import os
import re
import json
from pathlib import Path

class IO_Utils:

	def __init__(self):

		with open('config.json', "r", encoding="utf-8") as f:
			# Load the JSON data into a Python dictionary
			config = json.load(f)

		self.store_path= config['data_dir']

	def list_files(self,folder):
		return os.list

	def load_text_rows(self,path):

		# returns a list of lines for a text file

		text_rows = []
		with open(path,'r') as file:
			for row in file.readlines():
				if(len(row.strip().strip('\n'))>0):
					text_rows.append(row.replace('\n',''))

		return text_rows
