import pandas as pd
class dfPubMed():
  """This function return available dataframe for Pubmed datasef"""
  def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

  def data_processer(self,path):
    """this is for the first compilation"""
    def reader(path):
      with open(path,"r") as files:
        text = files.read()
      return text
    text = reader(path)
    abstracts_list = list(filter(None, text.split("###")))
    id_list = []
    dic_list = []
    lens_abs = []

    for abstract in abstracts_list:
        order = 0
        abstract_lines = abstract.splitlines()
        lens_abs.append(len(abstract_lines))
        for line in abstract_lines:
            if line.isdigit():
                id_list.append(line)
            elif len(line)==0:
              pass
            else:
                sentence = line.split("\t")
                if len(sentence) > 1:
                  dic = {"id":id_list[-1],"type": sentence[0], "sentence": sentence[1], "order": abstract_lines.index(line), "length":len(abstract_lines) - 2}
                  dic_list.append(dic)
                  order += 1
    return pd.DataFrame(dic_list)

  def get_train_data(self):
        """Returns the processed training data for class"""
        return self.data_processer(self.train_path)

  def get_tes_data(self):
        """Returns the processed training data for class"""
        return self.data_processer(self.train_path)
  def get_dataframe(self,data_path):
        """Returns the processed training dataframe for experimenter"""
        return self.data_processer(data_path)
  def plot_data_length_histogram(self, data_path):
        """Plots histogram of lengths of data"""
        data = self.data_processer(data_path)
        data['length'].plot.hist()
        plt.xlabel('Length')
        plt.ylabel('Frequency')
        plt.title('Histogram of Lengths of Data')
        plt.show()

