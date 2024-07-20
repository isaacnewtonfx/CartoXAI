from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

def subset_observation_data(obs, relevant_features, target):
    """
    A function to subset the observed dataset using the relevant feature names.

    Parameters
    ----------

    obs: geopandas dataframe
        The cleaned observed dataset which was an output from the exploratory data analysis step.
    relevant_features: List
        The list of relevant feature names that was an output from the exploratory data analysis step.
    target: str
        The name of target variable or feature which is also a column on the observed dataset. 

    Returns
    -------
    A tuple containing data on the subsetted observed dataset using the relevant features, and the target feature.  
    
    """

    data_features = obs[relevant_features]
    data_target   = obs[target]

    return (data_features, data_target)

def train_test_splitting(data_features, data_target, test_size=0.2):
    """
    A function to split the dataset into train and test sets for the purpose of machine learning.

    Parameters
    ----------

    data_features: pandas dataframe
        The subsetted observed dataset using the relevant features.
    data_target: pandas Series
        The target variable on the subsetted observed dataset.
    test_size: float
        The size of the test set. Ranges from 0 to 1 were 1 is 100% (default: 0.2)

    Returns
    -------

    A tuple containing the splited datasets in the order of Train Features Data, Test Features Data, Train Target Data, Test Target Data
    
    """

    X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, test_size=test_size, random_state=42)
    return (X_train, X_test, y_train, y_test)


def define_model_pipeline(preprocessor, processor):
    """
    A function to create a chain of operation in the order of preprocessing the features for a downstream machine learning model.

    Parameters
    ----------

    preprocessor: StandardScaler
        A numeric proeprocessing object from the sklearn.preprocessing package for scaling the numeric feature values.
    processor: RandomForestRegressor
        A random forest machine learning regressor from the sklearn.ensemble package.
    
    Returns
    -------

    A sklearn model pipeline object containing the preprocessor and the random forest regressor.
    
    """

    model = make_pipeline(preprocessor, processor)
    return model