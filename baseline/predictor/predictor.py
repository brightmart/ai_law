import json
import thulac
from sklearn.externals import joblib



class Predictor(object):
	def __init__(self):
		self.tfidf = joblib.load('predictor/model/tfidf.model')
		self.law = joblib.load('predictor/model/law.model')
		self.accu = joblib.load('predictor/model/accu.model')
		self.time = joblib.load('predictor/model/time.model')
		self.batch_size = 1
		
		self.cut = thulac.thulac(seg_only = True)

	def predict_law(self, vec):
		y = self.law.predict(vec)
		return [y[0] + 1]
	
	def predict_accu(self, vec):
		y = self.accu.predict(vec)
		return [y[0] + 1]
	
	def predict_time(self, vec):

		y = self.time.predict(vec)[0]
		
		#返回每一个罪名区间的中位数
		if y == 0:
			return -2
		if y == 1:
			return -1
		if y == 2:
			return 120
		if y == 3:
			return 102
		if y == 4:
			return 72
		if y == 5:
			return 48
		if y == 6:
			return 30
		if y == 7:
			return 18
		else:
			return 6
		
	def predict(self, content):
		fact = self.cut.cut(content[0], text = True)
		
		vec = self.tfidf.transform([fact])
		ans = {}

		ans['accusation'] = self.predict_accu(vec)
		ans['articles'] = self.predict_law(vec)
		ans['imprisonment'] = self.predict_time(vec)
		
		print(ans)
		return [ans]

		 
