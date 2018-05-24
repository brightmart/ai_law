import json


def init():

	f = open('law.txt', 'r', encoding = 'utf8')
	law = {}
	lawname = {}
	line = f.readline()
	while line:
		lawname[len(law)] = line.strip()
		law[line.strip()] = len(law)
		line = f.readline()
	f.close()


	f = open('accu.txt', 'r', encoding = 'utf8')
	accu = {}
	accuname = {}
	line = f.readline()
	while line:
		accuname[len(accu)] = line.strip()
		accu[line.strip()] = len(accu)
		line = f.readline()
	f.close()


	return law, accu, lawname, accuname


law, accu, lawname, accuname = init()


def getClassNum(kind):
	global law
	global accu

	if kind == 'law':
		return len(law)
	if kind == 'accu':
		return len(accu)


def getName(index, kind):
	global lawname
	global accuname
	if kind == 'law':
		return lawname[index]
		
	if kind == 'accu':
		return accuname[index]
	

def gettime(time):
	#将刑期用分类模型来做
	v = int(time['imprisonment'])

	if time['death_penalty']:
		return 0
	if time['life_imprisonment']:
		return 1
	elif v > 10 * 12:
		return 2
	elif v > 7 * 12:
		return 3
	elif v > 5 * 12:
		return 4
	elif v > 3 * 12:
		return 5
	elif v > 2 * 12:
		return 6
	elif v > 1 * 12:
		return 7
	else:
		return 8


def getlabel(d, kind):
	global law
	global accu
	
	# 做单标签

	if kind == 'law':
		# 返回多个类的第一个
		return law[str(d['meta']['relevant_articles'][0])]
	if kind == 'accu':
		return accu[d['meta']['accusation'][0]]
		
	if kind == 'time':
		return gettime(d['meta']['term_of_imprisonment'])
	
	return label
	


