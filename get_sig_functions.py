import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import iisignature as isig
import matplotlib.dates as mdates
import matplotlib.patches as patches
import datetime as dt


# read_airplane ="C:/Users/ecoramy/my_wdc/Ramy_codes/data/airplane.csv"
# data0 =pd.read_csv(read_airplane)

# # read_aapl ="C:/Users/ecoramy/my_wdc/Polymodel-v3/AAPL.csv"
# # data =pd.read_csv(read_aapl, index_col='Time')
levels = pd.read_csv('AAPL4PATH.csv', index_col='Time')
levels.index = pd.to_datetime( levels.index )
signal = levels[['IVV', 'AAPL','pb','pe1', 'USAFINF']];
signal_path = signal.to_numpy()

date_start = dt.datetime(2014, 12, 31)
date_end = dt.datetime(2021, 3, 31)

# print(data)

# X =data.iloc[:,1:3].values
# x =data.iloc[:,1:2].values #or  x = pd.DataFrame(data.iloc[:, 1])
# y =data.iloc[:,2].values #or y = data.iloc[:, -1:]

def get_returns(c):
    return np.diff(np.log(c))

def time_embedding(rawdata):
    """Read one-sample input, n-dimensional vector :=Matrix and outputs its path with the embedding selected

		Parameters
		----------
    		- rawdata datatype is DataFrame. 
			- If data='...', 
			- If data='...' or '...', 

		Returns
		-------
		signal_path: array of float64, shape (n_points,d) 
			Array with points in R^d that should be linearly interpolated to get
			the embedding.
		"""
    signal =rawdata

    n_points =np.shape(signal)[0]
    
    signal_path =np.hstack((signal,np.linspace(0,1,num=n_points).reshape((n_points,1))))       
            
    return(signal_path)


def insert_zero_at_beginning(arr):
    # Get the number of columns
    num_cols = arr.shape[1]
    # Create a row of zeros
    zero_row = np.zeros((1, num_cols), dtype=np.float64)
    # Insert the row of zeros at the beginning of the array
    new_arr = np.vstack((zero_row, arr))
    return new_arr


def basepoint_time_embedding(rawdata):
    """Read one-sample input, n-dimensional vector :=Matrix and outputs its path with the embedding selected

		Parameters
		----------
		file: str
			- If data='quick_draw', it is the string containing the raw drawing
			coordinates.
			- If data='urban_sound' or 'motion_sense', it is the path to the
			sample file.

		Returns
		-------
		path: array, shape (n_points,d)
			Array with points in R^d that should be linearly interpolated to get
			the embedding.
		"""
    signal =rawdata

    n_points =np.shape(signal)[0]
    signal_path_pre =np.hstack((signal,np.linspace(0,1,num=n_points).reshape((n_points,1))))       
    signal_path =insert_zero_at_beginning(signal_path_pre)
            
    return(signal_path)


def leadlag_embedding(rawdata,ll):
    
 	""" Return lead lag embedding from initial path.

 	Parameters
 	----------
 	path: array, shape (n_points,d) - rawdata datatype is DataFrame. 
		The array storing the n_points coordinates in R^d that constitute a 
		piecewise linear path.        

 	ll: int
		The number of lags.

 	Returns
 	-------
 	signal_path: array of float, shape (n_points+ll,ll*d +1)
		The array storing the coordinates in R^d that constitute the
		lead lag embedding of the initial path.
 	"""
    
 	path  =rawdata.to_numpy()
 	last_values=path[-1,:]
 	path=np.vstack((path,np.repeat([last_values],ll,axis=0)))
 	n_points=np.shape(path)[0]
 	dim=np.shape(path)[1]
 	for k in range(ll):
         for i in range(dim):
             path=np.hstack((path,np.zeros((n_points,1))))
             path[k+1:,-1]=path[:-k-1,i]
 	
 	signal_path =np.hstack((path,np.linspace(0,1,num=n_points).reshape((n_points,1))))
 	return(signal_path)


def path_to_sig(path, order):
		"""Read one sample and output its signature with the embedding selected
		by self.embedding.

		Parameters
		----------
		file: str
			- If data='quick_draw', it is the string containing the raw drawing
			coordinates.
			- If data='urban_sound' or 'motion_sense', it is the path to the
			sample file.

		Returns
		-------
		sig: array, shape (p)
			Array containing signature coefficients computed on the embedded
			path of the sample corresponding to file.
		"""
		
		sig =isig.sig(path,order)
		return(sig)

def path_to_dyadic_sig(path,order,dyadic_level, sig_dimension):
		"""Read one sample and output a vector containing a concatenation of
		signature coefficients. The path is divided into 2^dyadic_levelsubpaths 
		and a signature vector is computed on each subpath. All vectors
		obtained in this way are then concatenated.

		Parameters
		----------
		file: str
			- If data='quick_draw', it is the string containing the raw drawing
			coordinates.
			- If data='urban_sound' or 'motion_sense', it is the path to the
			sample file.

		dyadic_level: int
			It is the level of dyadic partitions considered. The path is divided
			into 2^dyadic_level subpaths and signatures are computed on each
			subpath.

		Returns
		-------
		sig: array, shape (p)
			A signature vector containing all signature coefficients. It is of
			shape p=2^dyadic_level*self.get_sig_dimension().
		"""
# 		path=self.data_to_path(file)
		n_points=np.shape(path)[0]
		n_subpaths=2**dyadic_level
# 		window_size=n_points//n_subpaths
		if n_points //n_subpaths ==1:
			window_size =( n_points//n_subpaths ) +1
		else:
			window_size =( n_points//n_subpaths )
		
		if n_subpaths>n_points:
			path=np.vstack(
				(path,np.zeros((n_subpaths-n_points,np.shape(path)[1]))))
			window_size=1 +1
		siglength=  int(sig_dimension )#isig.siglength(len(path[0]), order)  #self.get_sig_dimension()
		sig=np.zeros(n_subpaths*siglength)
		for i in range(n_subpaths):
			if i==n_subpaths-1:
				subpath=path[i*window_size:,:]   #subpath=path[i*window_size]
			else:
				subpath=path[i*window_size:(i+1)*window_size,:]  #subpath=path[i*window_size:(i+1)*window_size]
			sig[i*siglength:(i+1)*siglength]=isig.sig(subpath,order)
		return(sig)

def create_rectilinear(path): # path of type array
	""" Return rectilinear embedding from an initial path.

	Parameters
	----------
	path: array, shape (n_points,d)
		The array storing n_points coordinates in R^d that constitute a 
		piecewise linear path.

	Returns
	-------
	new_path: array shape ((n_points-1)*d+1,d)
		The array storing the coordinates in R^d that constitute the
		rectilinear embedding of the initial path.
	"""
	n_points=np.shape(path)[0]
	dim=np.shape(path)[1]
	new_path=np.zeros(((n_points-1)*dim+1,dim))
	new_path[0,:]=path[0,:]
	for i in range(1,n_points):
		new_path[(i-1)*dim+1:i*dim+1,:]=path[i-1,:]
		for j in range(dim):
			new_path[(i-1)*dim+j+1:i*dim+1,j]=path[i,j]
	return(new_path)

def duplicate_observations(data):
    """ Return rectilinear embedding from an initial path.

	Parameters
	----------
	path: 1-dime list, shape (n_points,1)		

	Returns
	-------
	results: duplicated list (2*n_points -2,1)	
		
	"""
    # Exclude first and last elements, then duplicate the middle part
    middle_data = data[1:-1]
    duplicated_data = np.repeat(middle_data, 2)
    
    # Concatenate the first element, duplicated middle part, and last element
    result = np.concatenate(([data[0]], duplicated_data, [data[-1]]))
    return result

def plot_duplicated_data(data, duplicated_data):
    plt.figure(figsize=(10, 6))
    plt.plot(data, label="Original Data", marker='o', linestyle='-', color='blue')
    plt.plot(duplicated_data, label="Duplicated Data", marker='x', linestyle='--', color='red')
    plt.title("Original Data vs Duplicated Data")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

def leadlag(X):
    '''
    Returns lead-lag-transformed stream of X (1-dim)

    Arguments:
        X: list, whose elements are tuples of the form
        (time, value).

    Returns:
        list of points on the plane, the lead-lag
        transformed stream of X
    '''

    l=[]

    for j in range(2*(len(X))-1):
        i1=j//2
        i2=j//2
        if j%2!=0:
            i1+=1
        l.append((X[i1][1], X[i2][1]))

    return l

def plotLeadLag(X, diagonal=True):
    '''
    Plots the lead-laged transformed path X. If diagonal
    is True, a line joining the start and ending points
    is displayed.

    Arguments:
        X: list, whose elements are tuples of the form
        (X^lead, X^lag) diagonal: boolean, default is
        True. If True, a line joining the start and
        ending points is displayed.
    '''
    for i in range(len(X)-1):
        plt.plot([X[i][1], X[i+1][1]], [X[i][0], X[i+1][0]],
                    color='k', linestyle='-', linewidth=2)


    # Show the diagonal, if diagonal is true
    if diagonal:
        plt.plot([min(min([p[0] for p in X]), min([p[1]
                for p in X])), max(max([p[0] for p in X]),
                max([p[1] for p in X]))], [min(min([p[0]
                for p in X]), min([p[1] for p in X])),
                max(max([p[0] for p in X]), max([p[1] for
                p in X]))], color='r', linestyle='-',  #color='#BDBDBD'
                linewidth=1)

    axes=plt.gca()
    axes.set_xlim([min([p[1] for p in X])-1, max([p[1] for
                  p in X])+1])
    axes.set_ylim([min([p[0] for p in X])-1, max([p[0] for
                  p in X])+1])
    axes.get_yaxis().get_major_formatter().set_useOffset(False)
    axes.get_xaxis().get_major_formatter().set_useOffset(False)
    axes.set_aspect('equal', 'datalim')
    plt.show()
    
def timejoined(X):
    '''
    Returns time-joined transformation of the stream of
    data X (1-dim)

    Arguments:
        X: list, whose elements are tuples of the form
        (time, value).

    Returns:
        list of points on the plane, the time-joined
        transformed stream of X
    '''
    X.append(X[-1])
    l=[]

    for j in range(2*(len(X))+1+2):
            if j==0:
                    l.append((X[j][0], 0))
                    continue
            for i in range(len(X)-1):
                    if j==2*i+1:
                            l.append((X[i][0], X[i][1]))
                            break
                    if j==2*i+2:
                            l.append((X[i+1][0], X[i][1]))
                            break
    return l

def plottimejoined(X):
    '''
    Plots the time-joined transfomed path X.

    Arguments:
        X: list, whose elements are tuples of the form (t, X)
    '''

    for i in range(len(X)-1):
        plt.plot([X[i][0], X[i+1][0]], [X[i][1], X[i+1][1]],
                color='k', linestyle='-', linewidth=2)

        # print( [X[i][0], X[i+1][0]], [X[i][1], X[i+1][1]] )

    axes=plt.gca()
    axes.set_xlim( [min([p[0] for p in X]) , max([p[0] for p in X])+1] )
    
    axes.set_ylim(  [min([p[1] for p in X]) , max([p[1] for p in X])+1]  )
    
    axes.get_yaxis().get_major_formatter().set_useOffset(False)
    axes.get_xaxis().get_major_formatter().set_useOffset(False)
    # axes.set_aspect('equal', 'datalim')
    plt.xlabel('$X_{\u03C4}={\u03C4}$')
    plt.ylabel('$Y_{\u03C4} =food \ inflation$')
    plt.title('time-joined transformation')
    plt.show()
    
def listOfTuples(l1, l2):
    return list(map(lambda x, y:(x,y), l1, l2))

def expand_window(elements, window_size):
    """
    Generate expanding windows from a 2D Numpy array
      
    Parameters
    ----------
    elements : 2D NumPy array
        Input array with k columns.
    window_size : Int
        The initial fixed window size

    Yields
    ------
    windows: 2D NumPy array
        The expanding windows.

    """
    rows, cols =elements.shape
    if cols < 1:
        raise ValueError("Input array must have at least 1 col.")
        
    for i in range(window_size, rows +1):
        yield elements[:i, :]  


if __name__ == '__main__':
      data1 =levels['USAFINF'];
      date_index = [int(ts.to_pydatetime().timestamp())/((1*(10**7))/1.0) for ts in data1.index.tolist()]      
      x1 =listOfTuples(date_index, data1.tolist() )
      dataTJ = timejoined(x1)
      dataTJ.pop(0)
      plottimejoined(dataTJ)
      
      dataLL = leadlag(x1)
      plotLeadLag(dataLL)
      
      
      plt.figure(figsize = (20,10))
      plt.title('Lead Lag prices from {} to {}'.format(date_start,date_end))												
      plt.plot(dataLL)
      plt.show()
      
      data0 =signal_path[:,[4]]           
      z =time_embedding(data0)
      z[:,[0,1]] =z[:, [1, 0]]
      date_index_1 =z[:,0].tolist()
      data11 =z[:,1].tolist()
      
      x2 =listOfTuples(date_index_1, data11)
        # x2 =listOfTuples(date_index, data1.tolist() )    
      dataTJ1 =timejoined(x2)
      dataTJ1.pop(0)
      plottimejoined(dataTJ1)
      
      

      
      
# #     #z =data_to_path_emdedding('lead-lag', data0)
# #     # r =leadlag_embedding(np.array(signal_path ),ll=1)
# #     # s =path_to_sig(r, 2)
#       # z_rec =create_rectilinear(signal_path)
#       # w =duplicate_observations(data1.tolist())
#       data =np.random.rand(10,5)
#       window_size =3
      
#       for window in expand_window(signal_path, window_size):
#           print(window)
#           print("-"*20)
      
      