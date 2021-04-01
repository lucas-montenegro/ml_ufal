import requests as req
import json
import random

# parameters for ils-vnd
total_iterations = 50
current_iteration = 0
iterations_without_improvement = 15 # total of iterations to reinitialize the solution
alpha = 0 # variable related with iterations_without_improvement 
neighbors_structures = 2

best_solution = -1000000
current_solution = -1000000
test_solution = -1000000

angle_increase = 10
angle_decrease = 5

angle_increase2 = 45
angle_decrease2 = 20

# storing the best angles for phi and theta
best_phi = [0, 0, 0]
best_theta = [0, 0, 0]

# storing the current angles for phi and theta
current_phi = [0, 0, 0]
current_theta = [0, 0, 0]

# storing the testing angles for phi and theta
test_phi = [0, 0, 0]
test_theta = [0, 0, 0]

# setting random seed with the system time
random.seed()

def update_best():
    global best_solution, current_solution, best_phi, best_theta

    for i in [0, 1, 2]:
        best_theta[i] = current_theta[i]
        best_phi[i] = current_phi[i] 

    best_solution = current_solution

def update_result():
    global test_solution
    response = req.get('https://aydanomachado.com/mlclass/02_Optimization.php?phi1=' + str(test_phi[0]) + '&theta1=' + str(test_theta[0]) + '&phi2=' + str(test_phi[1]) + '&theta2=' + str(test_theta[1]) + '&phi3=' + str(test_phi[2]) + '&theta3=' + str(test_theta[2]) + '&dev_key=LW')
    solution = json.loads(response.text)['gain']
    test_solution = solution

    #response = req.get('http://localhost:8080/antenna/simulate?phi1=' + str(test_phi[0]) + '&theta1=' + str(test_theta[0]) + '&phi2=' + str(test_phi[1]) + '&theta2=' + str(test_theta[1]) + '&phi3=' + str(test_phi[2]) + '&theta3=' + str(test_theta[2]))
    #solution = response.text.split('\n')
    #test_solution = float(solution[0])

def update_angles():
    global test_phi, test_theta

    for i in [0, 1, 2]:
        current_theta[i] = test_theta[i]
        current_phi[i] = test_phi[i]

def mutation(): # perturbação
    global test_phi, test_theta, current_solution, test_solution
    '''
    index_1 = random.randrange(0, 3, 1)
    index_2 = random.randrange(0, 3, 1)

    test_phi[index_1] = random.randrange(0, 360, 1)
    test_theta[index_2] = random.randrange(0, 360, 1)
    '''
    operation = random.randrange(0, 2, 1)

    if(operation == 0):
        for i in [0, 1, 2]:
            test_phi[i] = random.randrange(0, 360, 1)
    else:
        for i in [0, 1, 2]:
            test_theta[i] = random.randrange(0, 360, 1)

    update_result()

    # update the current solution if its worse than the test_solution
    if(current_solution <= test_solution):
        update_angles()
        current_solution = test_solution

def decrease_phi_or_theta(angle):
    global current_solution, test_solution, angle_increase2
    
    old_angles = [0, 0, 0]

    for i in [0, 1, 2]:
        old_angles[i] = angle[i]
        if((angle[i] - angle_decrease) < 0):
            angle[i] = (angle[i] - angle_decrease) + 360
        else:
            angle[i] -= angle_decrease

    update_result()

    if(test_solution >= current_solution):    
        update_angles()
        current_solution = test_solution
        return True
    else:
        for i in [0, 1, 2]:
            angle[i] = old_angles[i]

        return False

def increase_phi_or_theta(angle):
    global current_solution, test_solution, angle_increase2
    
    old_angles = [0, 0, 0]

    for i in [0, 1, 2]:
        old_angles[i] = angle[i]
        angle[i] = (angle[i] + angle_increase2) % 360

    update_result()

    if(test_solution >= current_solution):    
        update_angles()
        current_solution = test_solution
        return True
    else:
        for i in [0, 1, 2]:
            angle[i] = old_angles[i]

        return False

def decrease_angle(angle, index):
    global current_solution, test_solution, angle_decrease

    angle_before_update = angle[index]
    if((angle[index] - angle_decrease) < 0):
        angle[index] = (angle[index] - angle_decrease) + 360
    else:
        angle[index] -= angle_decrease

    update_result()

    if(test_solution >= current_solution):    
        update_angles()
        current_solution = test_solution
        return True
    else:
        angle[index] += angle_decrease
        return False

def increase_angle(angle, index):
    global current_solution, test_solution, angle_increase
    
    angle_before_update = angle[index]
    angle[index] = (angle[index] + angle_increase) % 360

    update_result()

    if(test_solution > current_solution):    
        update_angles()
        current_solution = test_solution
        return True
    else:
        angle[index] = angle_before_update
        return False


def generate_random_solution():
    global test_phi, test_theta, current_solution, test_solution
    for i in [0, 1, 2]:
            test_theta[i] = random.randrange(0, 360, 1)
            test_phi[i] = random.randrange(0, 360, 1)

    update_result()

    # update the current solution if its worse than the test_solution
    if(current_solution <= test_solution):
        update_angles()
        current_solution = test_solution


def local_search():
    global test_phi, test_theta
    K = 0 # variable related with neighbors_structures
    angle = 0
    index = 0
    operation = 0
    angle_list = [test_phi, test_theta]
    
    while(True):
        if(K == 0):
            angle = random.randrange(0, 2, 1)
            operation = random.randrange(0, 2, 1)
            index = random.randrange(0, 3, 1)

            if(operation == 0):
                if(increase_angle(angle_list[angle], index) == True):
                    K = 0
                else:
                    K += 1         
            if(operation == 1):
                if(decrease_angle(angle_list[angle], index) == True):
                    K = 0
                else:
                    K += 1         
        elif(K == 1):   
            angle = random.randrange(0, 2, 1)
            operation = random.randrange(0, 2, 1)

            if(operation == 0):
                if(increase_phi_or_theta(angle_list[angle]) == True):
                    K = 0
                else:
                    K += 1      
            if(operation == 1):
                if(increase_phi_or_theta(angle_list[angle]) == True):
                    K = 0
                else:
                    K += 1       
        else:
            return
        
def ils_vnd():
    global current_iteration, total_iterations, alpha, current_solution, best_solution
    generate_random_solution()

    while(current_iteration < total_iterations):
        if(alpha == 0):
            generate_random_solution()
            alpha = iterations_without_improvement # reinitializes alpha 
            
        local_search() # apply the local search

        if(best_solution < current_solution):
            update_best()
        else:
            alpha -= 1

        mutation()

        current_iteration += 1



ils_vnd()
print(best_solution)
print(best_phi)
print(best_theta)