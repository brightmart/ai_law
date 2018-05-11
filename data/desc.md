#法律数据集

###文件组成
**data_train.json**: 15.5w  
**data_valid.json**: 1.7w  
**data_test.json**: 3.3w  

###数据组成
数据中涉及 **183个法条**、**202个罪名**，均为刑事案件  

###数据清洗
数据中筛除了刑法中前101条(前101条并不涉及罪名)，并且为了方便进行模型训练，将罪名和法条数量少于30的类删去。

###数据格式
数据利用json格式储存，每一行为一条数据，每条数据均为一个字典
#####字段及意义
* **fact**: 事实描述  
* **meta**: 标注信息，标注信息中包括:   
	* **criminals**: 被告(数据中均只含一个被告)  
	* **punish\_of\_money**: 罚款(单位：元)
	* **accusation**: 罪名  
	* **relevant\_articles**: 相关法条  
	* **term\_of\_imprisonment**: 刑期  
		刑期格式(单位：月)
		* **death\_penalty**: 是否死刑  
		* **life\_imprisonment**: 是否无期
		* **imprisonment**: 有期徒刑刑期


