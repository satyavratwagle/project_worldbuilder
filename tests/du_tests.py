from utils.datastore_utils import DatastoreUtilities
import os
import json
from pathlib import Path

class DU_Tests:

    def __init__(self):

        with open('config.json', "r", encoding="utf-8") as f:
            config = json.load(f)


        self.du = DatastoreUtilities(config)
        self.du.load_embedding_model()

    def save_to_faiss(self):
        # WARNING : THIS TEST WILL OVERWRITE THE OLD DATASET. MAKE A COPY OF THE OLD DATASET IF NEEDED.


        dataset,index = self.du.update_faiss_dataset()    
        self.du.overwrite_faiss_dataset(dataset,index)
        self.du.reload_faiss_dataset()

    def restore_old_dataset(self):
        # If the old dataset was deleted by the test, use this to restore.

        # Delete old dataset if exists
        if os.path.exists(self.faiss_dataset_path):
            shutil.rmtree(self.faiss_dataset_path)

        # Rename temporary location
        os.rename(temp_dataset_path,self.faiss_dataset_path)
        os.rename(temp_faiss_path,self.faiss_index_path)

    def check_recent_graphs(self):

        files = self.du.get_most_current_graph()
        print(files)

    def add_test_node(self,node_name,node_type,node_summary):

        self.du.add_node(node_name,node_type,node_summary)
        #self.du.save_graph()

    def add_test_edge(self,head,tail,desc):

        self.du.add_edge(head,tail,desc)
        
    def save_graph(self):
        self.du.save_graph()

t = DU_Tests()

t.du.add_node("Satya","person",["Satyavrat Rajaram Wagle","Purdue University"])
t.du.add_node("Anu","person",["Anupreeta Agate","CMU"])
#t.du.save_graph()

#dataset,index = t.du.filter_dataset()
#t.du.overwrite_faiss_dataset(dataset,index)

# Rerun the script here - Checking if completely new nodes can be added.
#t.du.add_node("Seattle","location",["We vibin here","In US","This place aint so bad cuh.","Bruh wasdis"])
#t.du.add_node("China","location",["nihao xinga","we vibin here too"])
#print(t.du.knowledge_graph)
#t.du.save_graph()

#dataset,index = t.du.filter_dataset()
#t.du.overwrite_faiss_dataset(dataset,index)

# Rerun the script here - Checking if new added summaries to nodes are picked up
#t.du.add_node("Anu","person",["Dawg this suum new"])
#t.du.add_node("Seattle","person",["Blud we shortnin"])

#t.du.remove_node('india')
#t.du.save_graph()

dataset,index = t.du.filter_dataset()
t.du.overwrite_faiss_dataset(dataset,index)