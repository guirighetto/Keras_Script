# ExtractFeatWeight

OUTPUT FORMAT FILES:

   Diss = True: 

   		feat_1 feat_2 feat_3 ... feat_n class_1
        feat_1 feat_2 feat_3 ... feat_n class_2
                                              .
                                              .
                                              .                                         
        feat_1 feat_2 feat_3 ... feat_n class_n

   Diss = False (SVM FORMAT): 
   
   		class_1 1:feat_1 2:feat_2 3:feat_3 ... n:feat_n
		class_2 1:feat_1 2:feat_2 3:feat_3 ... n:feat_n
		                                              .
		                                              .
		                                              .                                        
		class_n 1:feat_1 2:feat_2 3:feat_3 ... n:feat_n


line 95 --> model.add(Dense(2048)) -- Number of features extracted

# About CNN
ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/

