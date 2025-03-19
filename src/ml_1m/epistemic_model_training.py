from python_classes import *
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    print('Hello Ruhani')
    data, spliter = init(sequence_length=100, target_length=1, seed=2022, k_fold=5)	
    print(data.num_users)
    between_users = {
        'nicest':{
            'threshold':7.50, 
            'isMost':True, 
            'split_array':data.x_avg_ratings,
            'isModelTrained': False,
            'isEvaluationMetricsCalculated': False,
            'isGraphsPloted':False,
            'x_label':'Average Value of Ratings',
        },
                 
        'harshest':{
            'threshold':7.50, 
            'isMost':False, 
            'split_array':data.x_avg_ratings,
            'isModelTrained': False,
            'isEvaluationMetricsCalculated': False,
            'isGraphsPloted':False,
            'x_label':'Average Value of Ratings',
        },

        'most_active':{
            'threshold':100.00, 
            'isMost':True, 
            'split_array':data.x_n_ratings,
            'isModelTrained': False,
            'isEvaluationMetricsCalculated': False,
            'isGraphsPloted':False,
            'x_label':'Number of Rated Movies',
        },

        'least_active':{
            'threshold':100.00, 
            'isMost':False, 
            'split_array':data.x_n_ratings,
            'isModelTrained': False,
            'isEvaluationMetricsCalculated': False,
            'isGraphsPloted':False,
            'x_label':'Number of Rated Movies',
        },

        'most_typical':{
            'threshold':0.13, 
            'isMost':True, 
            'split_array':data.x_similarity,
            'isModelTrained': False,
            'isEvaluationMetricsCalculated': False,
            'isGraphsPloted':False,
            'x_label':'Average User Similarities to All Others',
        },

        'least_typical':{
            'threshold':0.13, 
            'isMost':False, 
            'split_array':data.x_similarity,
            'isModelTrained': False,
            'isEvaluationMetricsCalculated': False,
            'isGraphsPloted':False,
            'x_label':'Average User Similarities to All Others',
        },

        'most_consistent':{
            'threshold':4.00, 
            'isMost':True, 
            'split_array':data.x_var_ratings,
            'isModelTrained': False,
            'isEvaluationMetricsCalculated': False,
            'isGraphsPloted':False,
            'x_label':'None',
        },

        'least_consistent':{
            'threshold':4.00, 
            'isMost':False, 
            'split_array':data.x_var_ratings,
            'isModelTrained': False,
            'isEvaluationMetricsCalculated': False,
            'isGraphsPloted':False,
            'x_label':'None',
        },

        'most_open_minded':{
            'threshold':16, 
            'isMost':True, 
            'split_array':data.x_n_genres,
            'isModelTrained': False,
            'isEvaluationMetricsCalculated': False,
            'isGraphsPloted':False,
            'x_label':'None',
        },

        'least_open_minded':{
            'threshold':16, 
            'isMost':False, 
            'split_array':data.x_n_genres,
            'isModelTrained': False,
            'isEvaluationMetricsCalculated': False,
            'isGraphsPloted':False,
            'x_label':'None',
        },

    }
    
    between_user_data_reduction_techniques = [
		'most_open_minded', #[Done]
		'least_open_minded', #[Done]
		'most_consistent', #[Done]
		'least_consistent', #[Done]
		'most_active', #[Done]
		'least_active', #[Done]
		'nicest', #[Done]
		'harshest', #[Done]
		# 'most_typical', 
		# 'least_typical',
	]

    retained_user_parcentages = [
		1.00, 
		0.75, 
		0.50, 
		0.25, 
		0.00
	]
    model_saved_location = '../results/NextItNet/epistemic_uncertainty/'
	
    def get_check_point_file_name(fold, epoch, folder_path='valina_full_model/'):
        model_check_point_location = model_saved_location + folder_path + 'fold_%i_epoch_%i'%(fold, epoch)
        print('Model saved at:', model_check_point_location)
        return model_check_point_location
	
    # max_epochs = 5
    verbose = 1
    # batch_size = 512
    def model_train(model, train_set, validation_set, max_epochs, batch_size):
        model.fit(
			x = train_set.sequences, 
			y = train_set.targets,
			
			# epochs=max_epochs,
			verbose=verbose, 
			batch_size=batch_size,
			# validation_split = 0.20,
			validation_data=(
				validation_set.sequences, 
				validation_set.targets
			),
			validation_batch_size=batch_size,
			# callbacks = tf.keras.callbacks.EarlyStopping(),
		)
        return model
		
    def getHitRate(model, test_instances, T=1):
        m = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=20)
        hrT = np.zeros(T)
        for i in range(T):
            y_pred = model.predict(test_instances.sequences, verbose=0)
            m.update_state(test_instances.targets, tf.nn.softmax(y_pred[:, :-1]))
	    # print('HR@20:', m.result().numpy())
            hrT[i] = m.result().numpy()
        return np.mean(hrT)
    k_fold = 5
    max_epochs = 10
    batch_size = 512

    for between_user_technique in between_user_data_reduction_techniques:
        for retained_parcent in retained_user_parcentages:
            print('---------------------------------------------------------------------------------------')
            print('Technique:', between_user_technique, '--> Retained Parcentage:', retained_parcent)
            print('---------------------------------------------------------------------------------------')
            for fold in range(k_fold):
                print('Fold Id:', fold)
                train_set, validation_set, test_set = spliter.split_between_user_strategy(						        fold_id=fold, 
			        retained_parcent=retained_parcent,
	    		        name=between_user_technique, 
    				between_users=between_users
			)
							
                model_base = NextItNet(data=data, T=10).get_model()
                max_HR = 0

                for epochs in range(max_epochs):

                    model_base = model_train(model_base, train_set, validation_set, max_epochs, batch_size)

                    hr = getHitRate(model_base, test_set, T=10)
                    if hr > max_HR:
                        max_HR = hr
                        model_base.save_weights(model_saved_location + between_user_technique + '/%.2f/best_models/'%(retained_parcent) + 'fold_%i_best_model'%(fold))
                        print('model_saved')

                        print('Fold:', fold, ' Epochs:', epochs, ' Evaluation: HR@20: ', hr)

                    model_base.save_weights(get_check_point_file_name(fold, epochs, folder_path=between_user_technique + '/%.2f/'%(retained_parcent)))
    print('Exit')
