'''

This file contains utility functions for loading our data as well as utilities function to handle and process event dataframes. 

'''



def load_data(train_path='train10/train10/', test_path='test10/test10/'):
    
    '''
    Loads the data from the specified base path.
    
    Parameters:
        train_path (str): The base path to the training data.
        test_path (str): The base path to the testing data.
        
    Returns:
        x_train (list): A list of dataframes containing the training data.
        y_train (list): A list of labels for the training data.
    '''
    
    from tqdm import tqdm # for tracking progress
    import os
    import pandas as pd
    
    class_folders = ['Addition', 'Carnaval', 'Decider', 'Ecole', 'Fillette', 'Huitre', 'Joyeux', 'Musique', 'Pyjama', 'Ruisseau']

    # Initialize lists to hold dataframes and labels
    x_train = []
    x_test = []

    # Define the column names
    column_names = ['x', 'y', 'p', 't']

    # Loop over each class folder
    for folder in tqdm(class_folders):
        # Get the list of CSV files in this class folder
        csv_files = os.listdir(os.path.join(train_path, folder))
        
        # Loop over each CSV file
        for csv_file in csv_files:
            # Define the full path to the CSV file
            csv_path = os.path.join(train_path, folder, csv_file)
            
            # Load the CSV file into a dataframe with the specified column names and data types
            df = pd.read_csv(csv_path, names=column_names, header=None, dtype={'x': int, 'y': int, 'p': int, 't': int}, skiprows=1)
            
            # Append the dataframe to the list
            x_train.append(df)
            
    print('training data loaded')
            
    # Get the list of CSV files in the base path
    csv_files = os.listdir(test_path)
    
    # Loop over each CSV file
    for csv_file in csv_files:
        # Define the full path to the CSV file
        csv_path = os.path.join(test_path, csv_file)
        
        # Load the CSV file into a dataframe with the specified column names and data types
        df = pd.read_csv(csv_path, names=column_names, header=None, dtype={'x': int, 'y': int, 'p': int, 't': int}, skiprows=1) 
        
        # Append the dataframe to the list
        x_test.append(df)
    
    print('test data loaded')
            
    y_train = [i for i in range(10) for _ in range(32)]

    return x_train, x_test, y_train


def events_to_image(df, x_max=480, y_max=640, rotate=255):
    
    '''
    
    Converts a dataframe of events to a 2D image that is a 2D histogram of the events. rotates the image by the 255 degrees.
    
    Args:
        df (pd.DataFrame): A dataframe of events.
        x_max (int): The maximum x coordinate, defaults to 480.
        y_max (int): The maximum y coordinate, defaults to 640.
        rotate (int): The value to rotate the image, defaults to 255.
        
    Returns:
        np.ndarray: A 2D image.
    
    '''
    
    import numpy as np
    
    # Create a 2D histogram of the event data
    hist, _, _ = np.histogram2d(df['x'], df['y'], bins=(x_max, y_max), weights=df['p'])

    # Normalize the histogram to the range [0, rotate]
    hist = 255 * (hist - np.min(hist)) / (np.max(hist) - np.min(hist))

    # Rotate the image by the value of the rotate parameter
    hist = np.rot90(hist, rotate)

    return hist.astype(np.uint8)

def event_agg_no_polarity(x, y, p, t, T_r=100000, M=640, N=480, rotation=255): # Doesn't take into account polarity
    
    '''
    Aggregate events into superframes.
    
    Args:
        x (np.array): x coordinates of events
        y (np.array): y coordinates of events
        p (np.array): polarity of events
        t (np.array): timestamp of events
        T_r (float): time interval of superframes, defaults to 100000.
        M (int): image length, defaults to 640.
        N (int): image width, defaults to 480.
        rotation (int): The value to rotate the frames, defaults to 255.
        
    Returns:
        superframes (np.array): superframes
    '''
    
    from scipy.ndimage import rotate
    import numpy as np
    from tqdm import tqdm
    
    T_seq = t.max()
    T_frames = int((T_seq // T_r)) + 1
    
    frames_0 = np.zeros((T_frames, M, N)) # polarity == 0
    frames_1 = np.zeros((T_frames, M, N)) # polarity == 1
    
    for i in tqdm(range(T_frames)):
        idx_0 = np.where((t >= i * T_r) & (t< (i+1) * T_r) & (p == 0))[0]
        if len(idx_0) > 0:
            frames_0[i] = np.bincount(N * x[idx_0] + y[idx_0], minlength = M * N).reshape(M, N)
        
        idx_1 = np.where((t >= i * T_r) & (t < (i+1) * T_r) & (p == 1))[0]
        if len(idx_1) > 0:
            frames_1[i] = np.bincount(N * x[idx_1] + y[idx_1], minlength = M * N).reshape(M, N)
    
    # Rotate the frames
    frames_0 = rotate(frames_0, rotation, axes=(1,2), reshape=False)
    frames_1 = rotate(frames_1, rotation, axes=(1,2), reshape=False)
    
    superframes = np.concatenate((frames_0, frames_1), axis = 0)
    print('generated superframes with size:', superframes.shape)
    return superframes


def event_agg_polarity(x, y, p, t, T_r=100000, M=640, N=480, rotation=255): # Takes into account polarity
    
    '''
    Aggregate events into superframes.
    
    Args:
        x (np.array): x coordinates of events
        y (np.array): y coordinates of events
        p (np.array): polarity of events
        t (np.array): timestamp of events
        T_r (float): time interval of superframes, defaults to 100000.
        M (int): image length, defaults to 640.
        N (int): image width, defaults to 480.
        rotation (int): The value to rotate the frames, defaults to 255.
        
    Returns:
        superframes_0 (np.array): superframes for polarity 0
        superframes_1 (np.array): superframes for polarity 1
    '''
    
    from scipy.ndimage import rotate
    import numpy as np
    from tqdm import tqdm
    
    T_seq = t.max()
    T_frames = int((T_seq // T_r)) + 1
    
    frames_0 = np.zeros((T_frames, M, N)) # polarity == 0
    frames_1 = np.zeros((T_frames, M, N)) # polarity == 1
    
    for i in tqdm(range(T_frames)):
        idx_0 = np.where((t >= i * T_r) & (t < (i+1) * T_r) & (p == 0))[0]
        if len(idx_0) > 0:
            frames_0[i] = np.bincount(N * x[idx_0] + y[idx_0], minlength = M * N).reshape(M, N)
        
        idx_1 = np.where((t >= i * T_r) & (t < (i+1) * T_r) & (p == 1))[0]
        if len(idx_1) > 0:
            frames_1[i] = np.bincount(N * x[idx_1] + y[idx_1], minlength = M * N).reshape(M, N)
    
    # Rotate the frames
    frames_0 = rotate(frames_0, rotation, axes=(1,2), reshape=False)
    frames_1 = rotate(frames_1, rotation, axes=(1,2), reshape=False)
    
    print('generated superframes with size:', frames_0.shape, 'and', frames_1.shape)
    return frames_0, frames_1

    
def decompose_events(event_df):
    
    '''
    
    Decompose the events dataframe into tuple of individual columns x, y, polarity, and timestamp.
    
    Args:
        test_data (pd.DataFrame): A dataframe of events.
        
    Returns:
        4-tuple: x (np.array): x coordinates of events,
                 y (np.array): y coordinates of events,
                 p (np.array): polarity of events,
                 t (np.array): timestamp of events  
    '''
    
    import numpy as np
    
    return (np.array(event_df['x'].values),
            np.array(event_df['y'].values),
            np.array(event_df['p'].values),
            np.array(event_df['t'].values))