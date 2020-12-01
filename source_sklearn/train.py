from __future__ import print_function

import argparse
import os
import pandas as pd

# sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. 
# from sklearn.externals import joblib
# Import joblib package directly
import joblib

from sklearn.svm import SVC

## TODO: Import any additional libraries you need to define a model


# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    
    parser.add_argument('--kernel', type=str, default='rbf', help='type of hyperplane used to separate the data (default: rbf)')
    parser.add_argument('-C', type=float, default=1., help='penalty parameter of the error term (default: 1.)')
    parser.add_argument('--gamma', type=str, default='scale', help='parameter for non linear hyperplanes (default: scale)')
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    ## --- Your code here --- ##

    ## TODO: Define a model 
    model = SVC(kernel=args.kernel, C=args.C, gamma = args.gamma)
    
    ## TODO: Train the model
    model.fit(train_x, train_y)
    
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'kernel': args.kernel,
            'C': args.C,
            'gamma': args.gamma
        }
        joblib.dump(model_info, f)
    
    ## --- End of your code  --- ##

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
