import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.interpolate




"""
This program as is requires the following assumptions:
    1) The data lies in [-1,1]
    2) The data is "well distributed" as in no obvious outliers
    3) The data is monotone increasing
    4) The data is nonempty
    5) The width of the intervals (delta_x) evenly divdes [-1,1]
"""

#filter: array(array X array), int, rational -> array(array X array)
#filter(data,n = 0, delta = 1/10) is the filter data for quasi-chebyshev points
#If no n value given, then returns the most number of points that
#are chebyshev distributed.
#If there is not n quasi-chebyshev points, return None

def filter(data,n = 0,delta = 1/10):
    num_points = data.size
    #remove tail ends from both ends (10% each side)
    sigma = int(0.1 * num_points)
    x_data = data[0][sigma:num_points - sigma]
    y_data = data[1][sigma:num_points-sigma]
    #invalid data input
    if x_data.size == 0 or y_data.size == 0:
        print('Data Size error')
        return None
    sorted_x = np.sort(x_data)
    a,b = sorted_x[0], sorted_x[-1]
    #map data to [-1,1]
    mapped_x = np.array([linear_map(x,a,b) for x in sorted_x])
    
    filtered_mapped_x = d1_filter(mapped_x,n)
    if filtered_mapped_x == []:
        return None
    filtered_x = np.array([inverse_linear_map(x,a,b) for x  in filtered_mapped_x])
    filtered_data = collect_data(filtered_x,x_data,y_data)
    return filtered_data

#collect_data: array, array, array -> array(array X array)
#collect_data(filtered_x,x_data,y_data) is the array containing the filtered x values
#with their cooresponding y values

def collect_data(filtered_x,x_data,y_data):
    filtered_y = []
    for x in filtered_x:
        try: 
            x_ind = (np.where(x_data == x)[0][0])
        except:
            index = np.argwhere(filtered_x == x)
            filtered_x = np.delete(filtered_x,index)
            continue
        filtered_y.append(y_data[x_ind])
    filtered_data = [filtered_x,np.array(filtered_y)]
    return np.array(filtered_data)



#d1-filter: array, n, float, -> array
#d1_filter(x,n,delta_x) is the filtered array of x

def d1_filter(x,n =0, delta_x = 1/10):
    histogram = np.histogram(x, bins = int(2/delta_x))
    #distribution of given data
    hist_x = histogram[0]
    #points for the invervals
    interval_values = histogram[1]
    #create variable probabilities
    prob = create_prob(delta_x)
    #determine minimum size of input n and if input is valid
    min_n = minimum_n(prob)
    
    if n < min_n and n != 0:
        print('n not large enough. Made n the minimum value required: %d' % (min_n))
        n = min_n

    #check if n is specified
    if n != 0:
        fixed_prob = np.multiply(prob,n)
        success = check_n(n,hist_x,fixed_prob)
        if not success:
            print("Error: no distribution for given n")
            return []
        #Data contains a Chebyshev Distribution for given n
        points = get_points(x,n,prob,interval_values)
        print('Distribution for given n found')
        return points

    
    m = x.size
    n = 0.5*m
    #determine which n is best
    success, best = determine(n,hist_x,prob,m)
    if success:
        #Pull out the n points from distribution
        points= get_points(x,best,prob,interval_values)
        print('Distribution found')
        return points
    else:
        print("Error: Chebyshev Distribution Not Found")
        return []

def minimum_n(prob):
    minimum = np.amin(prob)
    n = 1
    while minimum != 0:
        if minimum*n > 1:
            break
        n += 1
    return n


#check_n:int array,array -> bool
#Check if given n satisifies distribution size

def check_n(n,hist_x,fixed_prob):
    for i in range(hist_x.size):
        #If more points are required than exists in the interval return False
        if np.ceil(fixed_prob[i]) > hist_x[i]: return False
        return True


#determine:int,array,array,int,int,float -> bool, int
#Find largest n such that a chebyshev distribution can be found with n points

def determine(n,hist_x,prob, m, best = 0, t = 1,count = 1):
    #determine exact number required from intervals based on n
    count += 1
    #specific probability array based on fixed n
    fixed_prob = np.multiply(prob, n)
    #check the fixed n to see if this n is valid or not
    valid = check_n(n,hist_x,fixed_prob) 
    
    step = int((m-n)/2)
    if valid:
        best = n
        #step size too small to make noticable change
        if step <= t: return True, best
        else:
            return determine(n + step, hist_x, prob, m, best = best,count = count)
    if not valid:
        if best != 0: return True, best
        elif step <= t :return False, 0
        else:
            return determine(int(n/2), hist_x, prob, m, best = best,count=count)


#get_points: array, int, array -> array
#get_points(x,n,prob) is the filtered array based on the given optimal n

def get_points(x,n,prob,values):
    final = np.array([])
    #divide x into appropriate intervals
    intervals = make_intervals(x,prob,values)
    fixed_prob = np.multiply(prob,n)
    for i in range(intervals.size):
        #ideal number of points from the ith interval
        ideal_num = int(fixed_prob[i])
        #for each interval, pull out ideal number of points from middle of interval
        picked_points = pick_points(intervals[i],ideal_num)
        final = np.append(final,picked_points)
    return final


#make_intervals: array, array -> array
#make_intervals(x,prob) is x seperated into appropriate intervals

def make_intervals(x,prob,values):
    intervals = []
    for i in range(1,values.size):
        a = values[i-1]
        b = values[i]
        #pull out the x values in [a,b)
        interval = [v for v in x if (v >= a and v < b)]
        intervals.append(np.array(interval))
    return np.array(intervals)


#pick_points: array, int -> array
#pick_points(interval, ideal_num) is the array containing the ideal 
#number of "middle" points
def pick_points(interval, ideal_num):
    #array to store the "ideal" number of middle points from given interval
    ideal_points = np.array([])
    #add the endpoints of interval then remove them from the interval
    ideal_points = np.append(ideal_points,[interval[0],interval[-1]])
    interval = np.delete(interval,[0,interval.size - 1])
    for i in range(ideal_num):
        #select index closest to middle of interval
        n = int(np.floor(interval.size/2))
        ideal_points = np.append(ideal_points,interval[n])
        #update interval by removing chosen element to avoid repeating
        interval = np.delete(interval,n)
    return ideal_points

def pick_points(interval,ideal_num):
    ideal_points = np.array([])
    for i in range(ideal_num // 3):
        n = interval.size // 2
        ideal_points = np.append(ideal_points, [interval[0],interval[n],interval[-1]])
        interval = np.delete(interval,[0,n,interval.size - 1])
    return ideal_points


#norm is a norm. Thats it. I chose absolute value. Do what you want.
def norm(vec):
    return np.absolute(vec)

"""
create the probability vector for beta(1/2,1/2) [1 dimension]
"""
def cdf(x):
    pi = np.pi
    inpt = np.sqrt((x+1)/2)
    return 2/pi*(np.arcsin(inpt))

def create_prob(delta):
    prob = np.array([])
    #assume delta is a number that evenly diveds [-1,1]
    num_intervals = np.floor(int(2/delta))
    a = -1
    for i in range(1,int(num_intervals+1)):
        b = a + delta
        f_b, f_a = cdf(b), cdf(a)
        prob = np.append(prob, [f_b - f_a])
        a = b
    return prob

def linear_map(x,a,b):
    return 2*(x-a)/(b-a) - 1

def inverse_linear_map(x,a,b):
    return (x+1)*(b-a)/2 + a
