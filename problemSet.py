import numpy as np
import os
import pandas as pd
import ctypes
import copy
import re
from cec2013lsgo.cec2013 import Benchmark
from CEC2020.cec20_func import CEC2020_PROFILE

try:
    TORCH_FLAG = False
    # import torch
    # if torch.cuda.is_available():
    #     device = torch.device('cuda:0')
    #     TORCH_FLAG = True
    # else:
    #     TORCH_FLAG = False
except:
    TORCH_FLAG = False

def ray_f1(variables=None, problem_size=None):
    # Sphere
    variables = np.array(variables)
    # np.sum(variables**2)
    return np.sum(np.power(variables, 2))

def ray_f2(variables=None, problem_size=None):
    problem_size = len(variables)
    # t1 = 0.0
    # for i in range(problem_size - 1):
    #     t1 = t1 + np.power(variables[i] ** 2, variables[i + 1] ** 2 + 1) + np.power(variables[i+1] ** 2, variables[i] ** 2 + 1)
    
    variables = np.array(variables)
    variables_x = variables[:-1]
    variables_y = variables[1:]
    t1 = np.sum(np.power(np.power(variables_x, 2), np.power(variables_y, 2) + 1) +  np.power(np.power(variables_y, 2), np.power(variables_x, 2) + 1))
    return t1

def ray_f3(variables=None, problem_size=None):
    # High Conditioned Elliptic
    problem_size = len(variables)
    # tmp = 0
    # for i in range(problem_size):
    #     tmp += np.power(np.power(1000,i/(problem_size-1))*variables[i],2)

    variables = np.array(variables)
    w = np.arange(problem_size)
    tmp = np.sum(np.power(np.power(1000, w/(problem_size-1)) * np.array(variables) , 2))
    return tmp

def ray_f4(variables=None, problem_size=None):
    # Schwefel 2.21 Function
    problem_size = len(variables)
    variables = np.array(variables)
    return np.max(np.abs(variables))


def ray_f5(variables=None, problem_size=None):
    # Axis parallel hyper-ellipsoid function
    problem_size = len(variables)
    variables = np.array(variables)
    # tmp = 0
    # for i in range(problem_size):
    #     tmp += (i+1)*np.power(variables[i],2)
    w = np.arange(1, problem_size+1)
    tmp = np.sum(w * np.power(variables, 2))
    return tmp

def ray_f6(variables=None, problem_size=None):
    # Sum of Power
    problem_size = len(variables)
    variables = np.array(variables)
    # tmp = 0
    # for i in range(problem_size):
    #     tmp += np.power(np.absolute(variables[i]),i+2)

    w = np.arange(2, problem_size+2)
    tmp = np.sum(np.power(np.absolute(variables), w))
    return tmp

def ray_f7(variables=None, problem_size=None):
    # ZAKHAROV
    problem_size = len(variables)
    variables = np.array(variables)
    # tmp1 = 0
    # tmp2 = 0
    # for i in range(problem_size):
    #     tmp1 += np.power(variables[i],2)
    #     tmp2 += (i+1)*variables[i]

    tmp1 = np.sum(np.power(variables, 2))
    w = np.arange(1, problem_size+1)
    tmp2 = np.sum(w * np.array(variables))
    return tmp1+np.power(1/2*tmp2,2)+np.power(1/2*tmp2,4)

def ray_f8(variables=None, problem_size=None):
    # Rotated hyper-ellipsoid function
    problem_size = len(variables)
    variables = np.array(variables)
    # np.sum([np.power(np.sum([variables[j] for j in range(0, i)]), 2) for i in range(0, problem_size)])
    return np.sum([np.power(np.sum(variables[:i]), 2) for i in range(1, problem_size+1)])

def ray_f9(variables=None, problem_size=None):
    # Rastrigin function
    problem_size = len(variables)
    variables = np.array(variables)
    # np.sum(np.abs(np.power(variables, 2) - 10*np.cos(2*np.pi*variables) + 10))
    return np.sum(np.power(variables, 2) - 10*np.cos(2*np.pi*variables) + 10)

def ray_f10(variables=None, problem_size=None):
    # Ackley
    problem_size = len(variables)
    variables = np.array(variables)
    return -20*np.exp(-0.2*np.sqrt(np.sum(np.power(variables, 2))/problem_size))-np.exp(np.sum(np.cos(2*np.pi*variables))/problem_size)+20+np.exp(1)

def ray_f11(variables=None, problem_size=None):
   # Griewank function
    problem_size = len(variables)
    variables = np.array(variables)
    w = np.arange(1, problem_size+1)
    # w=[i for i in range(problem_size)]
    # w=[i+1 for i in w]
    return np.sum(np.power(variables, 2))/4000-np.prod(np.cos(variables/np.sqrt(w)))+1

def ray_f12(variables=None, problem_size=None):
    # STYBLINSKI-TANG FUNCTION
    # x∗=(−2.903534,…,−2.903534)

    problem_size = len(variables)
    variables = np.array(variables)
    # tmp1 = 0
    # for i in range(problem_size):
    #     tmp1 += np.power(variables[i], 4) - 16 * np.power(variables[i], 2) + 5 * variables[i]

    tmp1 = np.sum(np.power(variables, 4) - 16 * np.power(variables, 2) + 5 * variables)
    return tmp1/2


def ray_f13(variables=None, problem_size=None):
    # Csendes Function
    variables = np.array(variables)
    if np.prod(variables) == 0:
        res = 0
    else:
        res = (np.power(variables, 6) * (2 + np.sin(1/variables))).sum()
    return res

def ray_f14(variables=None, problem_size=None):
    # Xin-She Yang function 2
    problem_size = len(variables)
    # tmp1 = 0
    # tmp2 = 0
    # for i in range(problem_size):
    #     tmp1 += np.absolute(variables[i])
    #     tmp2 += np.sin(np.power(variables[i],2))

    variables = np.array(variables)
    tmp1 = np.sum(np.absolute(variables))
    tmp2 = np.sum(np.sin(np.power(variables, 2)))
    return tmp1 * np.exp(-tmp2)


def ray_f15(variables=None, problem_size=None):
    # Alpine Function No.01
    # problem_size = len(variables)
    # tmp = np.sum(np.abs(variables[i] * np.sin(variables[i]) + 0.1 * variables[i]) for i in range(problem_size))

    variables = np.array(variables)
    tmp = np.sum(np.abs(variables * np.sin(variables) + 0.1 * variables))
    return tmp


def ray_f16(variables=None, problem_size=None):
    # Michalewicz
    problem_size = len(variables)
    m = 10
    # tmp1 = 0
    # for i in range(problem_size):
    #     tmp1 += np.sin(variables[i]) * np.power(np.sin((i + 1) * np.power(variables[i], 2) / np.pi), 2 * m)

    variables = np.array(variables)
    w = np.arange(1, problem_size + 1)
    tmp1 = np.sum(np.sin(variables) * np.power(np.sin(w * np.power(variables, 2) / np.pi), 2 * m))
    return -tmp1


def rray_f1(variables):
  return ray_f1(variables)

def rray_f2(variables):
    return ray_f2(variables)

def rray_f3(variables):
  return ray_f3(variables)

def rray_f4(variables):
  return ray_f4(variables)

def rray_f5(variables):
  return ray_f5(variables)

def rray_f6(variables):
  return ray_f6(variables)

def rray_f7(variables):
  return ray_f7(variables)

def rray_f8(variables):
  return ray_f8(variables)

def rray_f9(variables):
  return ray_f9(variables)

def rray_f10(variables):
  return ray_f10(variables)

def rray_f11(variables):
  return ray_f11(variables)

def rray_f12(variables):
  return ray_f12(variables)

def rray_f13(variables):
  return ray_f13(variables)

def rray_f14(variables):
  return ray_f14(variables)

def rray_f15(variables):
  return ray_f15(variables)

def rray_f16(variables):
  return ray_f16(variables)

class CEC2014:
  def __init__(self, jc, dim):
    self.jc = jc
    self.dim = dim

  def CEC1(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,1)
  
  def CEC2(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,2)
  
  def CEC3(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,3)
  
  def CEC4(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,4)
  
  def CEC5(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,5)
  
  def CEC6(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,6)
  
  def CEC7(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,7)
  
  def CEC8(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,8)
  
  def CEC9(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,9)
  
  def CEC10(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,10)
  
  def CEC11(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,11)
  
  def CEC12(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,12)

  def CEC13(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,13)
  
  def CEC14(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,14)
  
  def CEC15(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,15)
  
  def CEC16(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,16)
  
  def CEC17(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,17)
  
  def CEC18(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,18)
  
  def CEC19(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,19)
  
  def CEC20(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,20)
  
  def CEC21(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,21)
  
  def CEC22(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,22)
  
  def CEC23(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,23)
  
  def CEC24(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,24)
  
  def CEC25(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,25)
  
  def CEC26(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,26)
  
  def CEC27(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,27)
  
  def CEC28(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,28)
  
  def CEC29(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,29)
  
  def CEC30(self, solution):
      return self.jc.test_func(solution.tolist(),[0],self.dim,1,30)


# {'func_name': 'cec1-2014', 'func': cec2014.CEC1, 'bound': [-100,100]},
# {'func_name': 'cec2-2014', 'func': cec2014.CEC2, 'bound': [-100,100]},
# {'func_name': 'cec3-2014', 'func': cec2014.CEC3, 'bound': [-100,100]},
# {'func_name': 'cec4-2014', 'func': cec2014.CEC4, 'bound': [-100,100]},
# {'func_name': 'cec5-2014', 'func': cec2014.CEC5, 'bound': [-100,100]},
# {'func_name': 'cec6-2014', 'func': cec2014.CEC6, 'bound': [-100,100]},
# {'func_name': 'cec7-2014', 'func': cec2014.CEC7, 'bound': [-100,100]},
# {'func_name': 'cec8-2014', 'func': cec2014.CEC8, 'bound': [-100,100]},
# {'func_name': 'cec9-2014', 'func': cec2014.CEC9, 'bound': [-100,100]},
# {'func_name': 'cec10-2014', 'func': cec2014.CEC10, 'bound': [-100,100]},
# {'func_name': 'cec11-2014', 'func': cec2014.CEC11, 'bound': [-100,100]},
# {'func_name': 'cec12-2014', 'func': cec2014.CEC12, 'bound': [-100,100]},
# {'func_name': 'cec13-2014', 'func': cec2014.CEC13, 'bound': [-100,100]},
# {'func_name': 'cec14-2014', 'func': cec2014.CEC14, 'bound': [-100,100]},
# {'func_name': 'cec15-2014', 'func': cec2014.CEC15, 'bound': [-100,100]},
# {'func_name': 'cec16-2014', 'func': cec2014.CEC16, 'bound': [-100,100]},
# {'func_name': 'cec17-2014', 'func': cec2014.CEC17, 'bound': [-100,100]},
# {'func_name': 'cec18-2014', 'func': cec2014.CEC18, 'bound': [-100,100]},
# {'func_name': 'cec19-2014', 'func': cec2014.CEC19, 'bound': [-100,100]},
# {'func_name': 'cec20-2014', 'func': cec2014.CEC20, 'bound': [-100,100]},
# {'func_name': 'cec21-2014', 'func': cec2014.CEC21, 'bound': [-100,100]},
# {'func_name': 'cec22-2014', 'func': cec2014.CEC22, 'bound': [-100,100]},
# {'func_name': 'cec23-2014', 'func': cec2014.CEC23, 'bound': [-100,100]},
# {'func_name': 'cec24-2014', 'func': cec2014.CEC24, 'bound': [-100,100]},
# {'func_name': 'cec25-2014', 'func': cec2014.CEC25, 'bound': [-100,100]},
# {'func_name': 'cec26-2014', 'func': cec2014.CEC26, 'bound': [-100,100]},
# {'func_name': 'cec27-2014', 'func': cec2014.CEC27, 'bound': [-100,100]},
# {'func_name': 'cec28-2014', 'func': cec2014.CEC28, 'bound': [-100,100]},
# {'func_name': 'cec29-2014', 'func': cec2014.CEC29, 'bound': [-100,100]},
# {'func_name': 'cec30-2014', 'func': cec2014.CEC30, 'bound': [-100,100]},


class CEC2019:
    def __init__(self):
        self.func_dict = {}
        self.load_func()

    def load_func(self):
        bench = Benchmark()
        num_of_func = bench.get_num_functions()
        self.func_dict = {}
        for i in range(1, num_of_func + 1):
            info = bench.get_info(i)
            # func = bench.get_function(i)
            # info['func'] = func
            self.func_dict[i] = info

class CEC2020:
    def __init__(self):
        self.cec2020_inst = CEC2020_PROFILE()

    def load_func(self, func_num):
        func_profile = self.cec2020_inst.load_func(func_num=func_num)
        return func_profile

class CEC2021:
    def __init__(self, mode, dim, c_entance_path):
        # Basic: B; Shift: S; Bias and Shift: BS; Shift and Rotation: SR; Bias, Shift and Rotation: BSR;
        # B, S, BS, SR, BSR
        self.mode = mode 
        self.dim = dim
        self.dim_c = ctypes.c_int(self.dim)
        self.lb = -100.0
        self.ub = 100.0
        self.c_entance_path = c_entance_path
        self.initial_cplus_func()
        self.func_dict = {
            # Bias
            'cecB1-2021': {'func_name': 'cecB1-2021', 'func': self.basic_f1, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecB2-2021': {'func_name': 'cecB2-2021', 'func': self.basic_f2, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecB3-2021': {'func_name': 'cecB3-2021', 'func': self.basic_f3, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecB4-2021': {'func_name': 'cecB4-2021', 'func': self.basic_f4, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecB5-2021': {'func_name': 'cecB5-2021', 'func': self.basic_f5, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecB6-2021': {'func_name': 'cecB6-2021', 'func': self.basic_f6, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecB7-2021': {'func_name': 'cecB7-2021', 'func': self.basic_f7, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecB8-2021': {'func_name': 'cecB8-2021', 'func': self.basic_f8, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecB9-2021': {'func_name': 'cecB9-2021', 'func': self.basic_f9, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecB10-2021': {'func_name': 'cecB10-2021', 'func': self.basic_f10, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            # Shift
            'cecS1-2021': {'func_name': 'cecS1-2021', 'func': self.shift_f1, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecS2-2021': {'func_name': 'cecS2-2021', 'func': self.shift_f2, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecS3-2021': {'func_name': 'cecS3-2021', 'func': self.shift_f3, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecS4-2021': {'func_name': 'cecS4-2021', 'func': self.shift_f4, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecS5-2021': {'func_name': 'cecS5-2021', 'func': self.shift_f5, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecS6-2021': {'func_name': 'cecS6-2021', 'func': self.shift_f6, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecS7-2021': {'func_name': 'cecS7-2021', 'func': self.shift_f7, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecS8-2021': {'func_name': 'cecS8-2021', 'func': self.shift_f8, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecS9-2021': {'func_name': 'cecS9-2021', 'func': self.shift_f9, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecS10-2021': {'func_name': 'cecS10-2021', 'func': self.shift_f10, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            # Bias and shift
            'cecBS1-2021': {'func_name': 'cecBS1-2021', 'func': self.biasShift_f1, 'bound': [self.lb, self.ub], 'optimal': 100, 'problem_size': self.dim, 'mode': self.mode},
            'cecBS2-2021': {'func_name': 'cecBS2-2021', 'func': self.biasShift_f2, 'bound': [self.lb, self.ub], 'optimal': 1100, 'problem_size': self.dim, 'mode': self.mode},
            'cecBS3-2021': {'func_name': 'cecBS3-2021', 'func': self.biasShift_f3, 'bound': [self.lb, self.ub], 'optimal': 700, 'problem_size': self.dim, 'mode': self.mode},
            'cecBS4-2021': {'func_name': 'cecBS4-2021', 'func': self.biasShift_f4, 'bound': [self.lb, self.ub], 'optimal': 1900, 'problem_size': self.dim, 'mode': self.mode},
            'cecBS5-2021': {'func_name': 'cecBS5-2021', 'func': self.biasShift_f5, 'bound': [self.lb, self.ub], 'optimal': 1700, 'problem_size': self.dim, 'mode': self.mode},
            'cecBS6-2021': {'func_name': 'cecBS6-2021', 'func': self.biasShift_f6, 'bound': [self.lb, self.ub], 'optimal': 1600, 'problem_size': self.dim, 'mode': self.mode},
            'cecBS7-2021': {'func_name': 'cecBS7-2021', 'func': self.biasShift_f7, 'bound': [self.lb, self.ub], 'optimal': 2100, 'problem_size': self.dim, 'mode': self.mode},
            'cecBS8-2021': {'func_name': 'cecBS8-2021', 'func': self.biasShift_f8, 'bound': [self.lb, self.ub], 'optimal': 2200, 'problem_size': self.dim, 'mode': self.mode},
            'cecBS9-2021': {'func_name': 'cecBS9-2021', 'func': self.biasShift_f9, 'bound': [self.lb, self.ub], 'optimal': 2400, 'problem_size': self.dim, 'mode': self.mode},
            'cecBS10-2021': {'func_name': 'cecBS10-2021', 'func': self.biasShift_f10, 'bound': [self.lb, self.ub], 'optimal': 2500, 'problem_size': self.dim, 'mode': self.mode},
            # Shift and rotation
            'cecSR1-2021': {'func_name': 'cecSR1-2021', 'func': self.shiftRotation_f1, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecSR2-2021': {'func_name': 'cecSR2-2021', 'func': self.shiftRotation_f2, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecSR3-2021': {'func_name': 'cecSR3-2021', 'func': self.shiftRotation_f3, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecSR4-2021': {'func_name': 'cecSR4-2021', 'func': self.shiftRotation_f4, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecSR5-2021': {'func_name': 'cecSR5-2021', 'func': self.shiftRotation_f5, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecSR6-2021': {'func_name': 'cecSR6-2021', 'func': self.shiftRotation_f6, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecSR7-2021': {'func_name': 'cecSR7-2021', 'func': self.shiftRotation_f7, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecSR8-2021': {'func_name': 'cecSR8-2021', 'func': self.shiftRotation_f8, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecSR9-2021': {'func_name': 'cecSR9-2021', 'func': self.shiftRotation_f9, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            'cecSR10-2021': {'func_name': 'cecSR10-2021', 'func': self.shiftRotation_f10, 'bound': [self.lb, self.ub], 'optimal': 0, 'problem_size': self.dim, 'mode': self.mode},
            # Bias, shift and rotation
            'cecBSR1-2021': {'func_name': 'cecBSR1-2021', 'func': self.biasShiftRotation_f1, 'bound': [self.lb, self.ub], 'optimal': 100, 'problem_size': self.dim, 'mode': self.mode},
            'cecBSR2-2021': {'func_name': 'cecBSR2-2021', 'func': self.biasShiftRotation_f2, 'bound': [self.lb, self.ub], 'optimal': 1100, 'problem_size': self.dim, 'mode': self.mode},
            'cecBSR3-2021': {'func_name': 'cecBSR3-2021', 'func': self.biasShiftRotation_f3, 'bound': [self.lb, self.ub], 'optimal': 700, 'problem_size': self.dim, 'mode': self.mode},
            'cecBSR4-2021': {'func_name': 'cecBSR4-2021', 'func': self.biasShiftRotation_f4, 'bound': [self.lb, self.ub], 'optimal': 1900, 'problem_size': self.dim, 'mode': self.mode},
            'cecBSR5-2021': {'func_name': 'cecBSR5-2021', 'func': self.biasShiftRotation_f5, 'bound': [self.lb, self.ub], 'optimal': 1700, 'problem_size': self.dim, 'mode': self.mode},
            'cecBSR6-2021': {'func_name': 'cecBSR6-2021', 'func': self.biasShiftRotation_f6, 'bound': [self.lb, self.ub], 'optimal': 1600, 'problem_size': self.dim, 'mode': self.mode},
            'cecBSR7-2021': {'func_name': 'cecBSR7-2021', 'func': self.biasShiftRotation_f7, 'bound': [self.lb, self.ub], 'optimal': 2100, 'problem_size': self.dim, 'mode': self.mode},
            'cecBSR8-2021': {'func_name': 'cecBSR8-2021', 'func': self.biasShiftRotation_f8, 'bound': [self.lb, self.ub], 'optimal': 2200, 'problem_size': self.dim, 'mode': self.mode},
            'cecBSR9-2021': {'func_name': 'cecBSR9-2021', 'func': self.biasShiftRotation_f9, 'bound': [self.lb, self.ub], 'optimal': 2400, 'problem_size': self.dim, 'mode': self.mode},
            'cecBSR10-2021': {'func_name': 'cecBSR10-2021', 'func': self.biasShiftRotation_f10, 'bound': [self.lb, self.ub], 'optimal': 2500, 'problem_size': self.dim, 'mode': self.mode},
        }
    
    def select_func(self, func_num):
        func_name = 'cec{}{}-2021'.format(self.mode, func_num)
        func_info = self.func_dict[func_name]
        func_info['bound'] = np.array(func_info['bound'])
        return func_info

    # fitness API for problems
    # Basic function
    def basic_f1(self, solution):
        # cec21_basic_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.basic_f1_cplus(xlst_p, self.dim_c)
        return fitness

    def basic_f2(self, solution):
        # cec21_basic_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.basic_f2_cplus(xlst_p, self.dim_c)
        return fitness

    def basic_f3(self, solution):
        # cec21_basic_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.basic_f3_cplus(xlst_p, self.dim_c)
        return fitness

    def basic_f4(self, solution):
        # cec21_basic_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.basic_f4_cplus(xlst_p, self.dim_c)
        return fitness

    def basic_f5(self, solution):
        # cec21_basic_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.basic_f5_cplus(xlst_p, self.dim_c)
        return fitness

    def basic_f6(self, solution):
        # cec21_basic_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.basic_f6_cplus(xlst_p, self.dim_c)
        return fitness

    def basic_f7(self, solution):
        # cec21_basic_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.basic_f7_cplus(xlst_p, self.dim_c)
        return fitness

    def basic_f8(self, solution):
        # cec21_basic_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.basic_f8_cplus(xlst_p, self.dim_c)
        return fitness

    def basic_f9(self, solution):
        # cec21_basic_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.basic_f9_cplus(xlst_p, self.dim_c)
        return fitness

    def basic_f10(self, solution):
        # cec21_basic_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.basic_f10_cplus(xlst_p, self.dim_c)
        return fitness

    # Shift function
    def shift_f1(self, solution):
        # cec21_shift_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.shift_f1_cplus(xlst_p, self.dim_c)
        return fitness

    def shift_f2(self, solution):
        # cec21_shift_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.shift_f2_cplus(xlst_p, self.dim_c)
        return fitness
    
    def shift_f3(self, solution):
        # cec21_shift_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.shift_f3_cplus(xlst_p, self.dim_c)
        return fitness
    
    def shift_f4(self, solution):
        # cec21_shift_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.shift_f4_cplus(xlst_p, self.dim_c)
        return fitness

    def shift_f5(self, solution):
        # cec21_shift_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.shift_f5_cplus(xlst_p, self.dim_c)
        return fitness

    def shift_f6(self, solution):
        # cec21_shift_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.shift_f6_cplus(xlst_p, self.dim_c)
        return fitness

    def shift_f7(self, solution):
        # cec21_shift_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.shift_f7_cplus(xlst_p, self.dim_c)
        return fitness

    def shift_f8(self, solution):
        # cec21_shift_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.shift_f8_cplus(xlst_p, self.dim_c)
        return fitness

    def shift_f9(self, solution):
        # cec21_shift_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.shift_f9_cplus(xlst_p, self.dim_c)
        return fitness

    def shift_f10(self, solution):
        # cec21_shift_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.shift_f10_cplus(xlst_p, self.dim_c)
        return fitness

    # Bias and shift
    def biasShift_f1(self, solution):
        # cec21_bias_shift_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.biasShift_f1_cplus(xlst_p, self.dim_c)
        return fitness

    def biasShift_f2(self, solution):
        # cec21_bias_shift_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.biasShift_f2_cplus(xlst_p, self.dim_c)
        return fitness

    def biasShift_f3(self, solution):
        # cec21_bias_shift_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.biasShift_f3_cplus(xlst_p, self.dim_c)
        return fitness

    def biasShift_f4(self, solution):
        # cec21_bias_shift_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.biasShift_f4_cplus(xlst_p, self.dim_c)
        return fitness

    def biasShift_f5(self, solution):
        # cec21_bias_shift_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.biasShift_f5_cplus(xlst_p, self.dim_c)
        return fitness

    def biasShift_f6(self, solution):
        # cec21_bias_shift_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.biasShift_f6_cplus(xlst_p, self.dim_c)
        return fitness

    def biasShift_f7(self, solution):
        # cec21_bias_shift_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.biasShift_f7_cplus(xlst_p, self.dim_c)
        return fitness


    def biasShift_f8(self, solution):
        # cec21_bias_shift_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.biasShift_f8_cplus(xlst_p, self.dim_c)
        return fitness

    def biasShift_f9(self, solution):
        # cec21_bias_shift_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.biasShift_f9_cplus(xlst_p, self.dim_c)
        return fitness

    def biasShift_f10(self, solution):
        # cec21_bias_shift_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.biasShift_f10_cplus(xlst_p, self.dim_c)
        return fitness

    # Shift and rotation
    def shiftRotation_f1(self, solution):
        # cec21_shift_rot_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.shiftRotation_f1_cplus(xlst_p, self.dim_c)
        return fitness

    def shiftRotation_f2(self, solution):
        # cec21_shift_rot_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.shiftRotation_f2_cplus(xlst_p, self.dim_c)
        return fitness

    def shiftRotation_f3(self, solution):
        # cec21_shift_rot_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.shiftRotation_f3_cplus(xlst_p, self.dim_c)
        return fitness

    def shiftRotation_f4(self, solution):
        # cec21_shift_rot_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.shiftRotation_f4_cplus(xlst_p, self.dim_c)
        return fitness

    def shiftRotation_f5(self, solution):
        # cec21_shift_rot_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.shiftRotation_f5_cplus(xlst_p, self.dim_c)
        return fitness

    def shiftRotation_f6(self, solution):
        # cec21_shift_rot_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.shiftRotation_f6_cplus(xlst_p, self.dim_c)
        return fitness

    def shiftRotation_f7(self, solution):
        # cec21_shift_rot_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.shiftRotation_f7_cplus(xlst_p, self.dim_c)
        return fitness

    def shiftRotation_f8(self, solution):
        # cec21_shift_rot_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.shiftRotation_f8_cplus(xlst_p, self.dim_c)
        return fitness

    def shiftRotation_f9(self, solution):
        # cec21_shift_rot_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.shiftRotation_f9_cplus(xlst_p, self.dim_c)
        return fitness

    def shiftRotation_f10(self, solution):
        # cec21_shift_rot_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.shiftRotation_f10_cplus(xlst_p, self.dim_c)
        return fitness

    # Bias, shift and rotation
    def biasShiftRotation_f1(self, solution):
        # cec21_bias_shift_rot_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.biasShiftRotation_f1_cplus(xlst_p, self.dim_c)
        return fitness
    
    def biasShiftRotation_f2(self, solution):
        # cec21_bias_shift_rot_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.biasShiftRotation_f2_cplus(xlst_p, self.dim_c)
        return fitness
    
    def biasShiftRotation_f3(self, solution):
        # cec21_bias_shift_rot_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.biasShiftRotation_f3_cplus(xlst_p, self.dim_c)
        return fitness

    def biasShiftRotation_f4(self, solution):
        # cec21_bias_shift_rot_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.biasShiftRotation_f4_cplus(xlst_p, self.dim_c)
        return fitness

    def biasShiftRotation_f5(self, solution):
        # cec21_bias_shift_rot_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.biasShiftRotation_f5_cplus(xlst_p, self.dim_c)
        return fitness

    def biasShiftRotation_f6(self, solution):
        # cec21_bias_shift_rot_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.biasShiftRotation_f6_cplus(xlst_p, self.dim_c)
        return fitness

    def biasShiftRotation_f7(self, solution):
        # cec21_bias_shift_rot_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.biasShiftRotation_f7_cplus(xlst_p, self.dim_c)
        return fitness

    def biasShiftRotation_f8(self, solution):
        # cec21_bias_shift_rot_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.biasShiftRotation_f8_cplus(xlst_p, self.dim_c)
        return fitness

    def biasShiftRotation_f9(self, solution):
        # cec21_bias_shift_rot_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.biasShiftRotation_f9_cplus(xlst_p, self.dim_c)
        return fitness

    def biasShiftRotation_f10(self, solution):
        # cec21_bias_shift_rot_func(solution, fitness, dim, pop_size, func_num)
        xlst = list(solution)
        xlst_c = (ctypes.c_double * self.dim)()
        xlst_c[:] = xlst
        xlst_p = ctypes.cast(xlst_c, ctypes.POINTER(ctypes.c_double))
        fitness = self.biasShiftRotation_f10_cplus(xlst_p, self.dim_c)
        return fitness

    def initial_cplus_func(self):
        p = ctypes.CDLL(self.c_entance_path)
        # Basic
        self.basic_f1_cplus = p.basic_f1_cplus
        self.basic_f1_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.basic_f1_cplus.restype = ctypes.c_double

        self.basic_f2_cplus = p.basic_f2_cplus
        self.basic_f2_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.basic_f2_cplus.restype = ctypes.c_double

        self.basic_f3_cplus = p.basic_f3_cplus
        self.basic_f3_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.basic_f3_cplus.restype = ctypes.c_double

        self.basic_f4_cplus = p.basic_f4_cplus
        self.basic_f4_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.basic_f4_cplus.restype = ctypes.c_double

        self.basic_f5_cplus = p.basic_f5_cplus
        self.basic_f5_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.basic_f5_cplus.restype = ctypes.c_double

        self.basic_f6_cplus = p.basic_f6_cplus
        self.basic_f6_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.basic_f6_cplus.restype = ctypes.c_double

        self.basic_f7_cplus = p.basic_f7_cplus
        self.basic_f7_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.basic_f7_cplus.restype = ctypes.c_double

        self.basic_f8_cplus = p.basic_f8_cplus
        self.basic_f8_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.basic_f8_cplus.restype = ctypes.c_double

        self.basic_f9_cplus = p.basic_f9_cplus
        self.basic_f9_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.basic_f9_cplus.restype = ctypes.c_double

        self.basic_f10_cplus = p.basic_f10_cplus
        self.basic_f10_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.basic_f10_cplus.restype = ctypes.c_double

        # Shift
        self.shift_f1_cplus = p.shift_f1_cplus
        self.shift_f1_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.shift_f1_cplus.restype = ctypes.c_double
        
        self.shift_f2_cplus = p.shift_f2_cplus
        self.shift_f2_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.shift_f2_cplus.restype = ctypes.c_double

        self.shift_f3_cplus = p.shift_f3_cplus
        self.shift_f3_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.shift_f3_cplus.restype = ctypes.c_double

        self.shift_f4_cplus = p.shift_f4_cplus
        self.shift_f4_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.shift_f4_cplus.restype = ctypes.c_double

        self.shift_f5_cplus = p.shift_f5_cplus
        self.shift_f5_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.shift_f5_cplus.restype = ctypes.c_double

        self.shift_f6_cplus = p.shift_f6_cplus
        self.shift_f6_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.shift_f6_cplus.restype = ctypes.c_double

        self.shift_f7_cplus = p.shift_f7_cplus
        self.shift_f7_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.shift_f7_cplus.restype = ctypes.c_double

        self.shift_f8_cplus = p.shift_f8_cplus
        self.shift_f8_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.shift_f8_cplus.restype = ctypes.c_double

        self.shift_f9_cplus = p.shift_f9_cplus
        self.shift_f9_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.shift_f9_cplus.restype = ctypes.c_double

        self.shift_f10_cplus = p.shift_f10_cplus
        self.shift_f10_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.shift_f10_cplus.restype = ctypes.c_double

        # Bias and shift
        self.biasShift_f1_cplus = p.biasShift_f1_cplus
        self.biasShift_f1_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.biasShift_f1_cplus.restype = ctypes.c_double

        self.biasShift_f2_cplus = p.biasShift_f2_cplus
        self.biasShift_f2_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.biasShift_f2_cplus.restype = ctypes.c_double

        self.biasShift_f3_cplus = p.biasShift_f3_cplus
        self.biasShift_f3_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.biasShift_f3_cplus.restype = ctypes.c_double

        self.biasShift_f4_cplus = p.biasShift_f4_cplus
        self.biasShift_f4_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.biasShift_f4_cplus.restype = ctypes.c_double

        self.biasShift_f5_cplus = p.biasShift_f5_cplus
        self.biasShift_f5_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.biasShift_f5_cplus.restype = ctypes.c_double

        self.biasShift_f6_cplus = p.biasShift_f6_cplus
        self.biasShift_f6_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.biasShift_f6_cplus.restype = ctypes.c_double

        self.biasShift_f7_cplus = p.biasShift_f7_cplus
        self.biasShift_f7_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.biasShift_f7_cplus.restype = ctypes.c_double

        self.biasShift_f8_cplus = p.biasShift_f8_cplus
        self.biasShift_f8_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.biasShift_f8_cplus.restype = ctypes.c_double

        self.biasShift_f9_cplus = p.biasShift_f9_cplus
        self.biasShift_f9_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.biasShift_f9_cplus.restype = ctypes.c_double

        self.biasShift_f10_cplus = p.biasShift_f10_cplus
        self.biasShift_f10_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.biasShift_f10_cplus.restype = ctypes.c_double

        # Shift and rotation
        self.shiftRotation_f1_cplus = p.shiftRotation_f1_cplus
        self.shiftRotation_f1_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.shiftRotation_f1_cplus.restype = ctypes.c_double

        self.shiftRotation_f2_cplus = p.shiftRotation_f2_cplus
        self.shiftRotation_f2_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.shiftRotation_f2_cplus.restype = ctypes.c_double

        self.shiftRotation_f3_cplus = p.shiftRotation_f3_cplus
        self.shiftRotation_f3_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.shiftRotation_f3_cplus.restype = ctypes.c_double

        self.shiftRotation_f4_cplus = p.shiftRotation_f4_cplus
        self.shiftRotation_f4_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.shiftRotation_f4_cplus.restype = ctypes.c_double

        self.shiftRotation_f5_cplus = p.shiftRotation_f5_cplus
        self.shiftRotation_f5_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.shiftRotation_f5_cplus.restype = ctypes.c_double

        self.shiftRotation_f6_cplus = p.shiftRotation_f6_cplus
        self.shiftRotation_f6_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.shiftRotation_f6_cplus.restype = ctypes.c_double

        self.shiftRotation_f7_cplus = p.shiftRotation_f7_cplus
        self.shiftRotation_f7_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.shiftRotation_f7_cplus.restype = ctypes.c_double

        self.shiftRotation_f8_cplus = p.shiftRotation_f8_cplus
        self.shiftRotation_f8_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.shiftRotation_f8_cplus.restype = ctypes.c_double

        self.shiftRotation_f9_cplus = p.shiftRotation_f9_cplus
        self.shiftRotation_f9_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.shiftRotation_f9_cplus.restype = ctypes.c_double

        self.shiftRotation_f10_cplus = p.shiftRotation_f10_cplus
        self.shiftRotation_f10_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.shiftRotation_f10_cplus.restype = ctypes.c_double

        # Bias, shift and rotation
        self.biasShiftRotation_f1_cplus = p.biasShiftRotation_f1_cplus
        self.biasShiftRotation_f1_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.biasShiftRotation_f1_cplus.restype = ctypes.c_double

        self.biasShiftRotation_f2_cplus = p.biasShiftRotation_f2_cplus
        self.biasShiftRotation_f2_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.biasShiftRotation_f2_cplus.restype = ctypes.c_double

        self.biasShiftRotation_f3_cplus = p.biasShiftRotation_f3_cplus
        self.biasShiftRotation_f3_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.biasShiftRotation_f3_cplus.restype = ctypes.c_double

        self.biasShiftRotation_f4_cplus = p.biasShiftRotation_f4_cplus
        self.biasShiftRotation_f4_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.biasShiftRotation_f4_cplus.restype = ctypes.c_double

        self.biasShiftRotation_f5_cplus = p.biasShiftRotation_f5_cplus
        self.biasShiftRotation_f5_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.biasShiftRotation_f5_cplus.restype = ctypes.c_double

        self.biasShiftRotation_f6_cplus = p.biasShiftRotation_f6_cplus
        self.biasShiftRotation_f6_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.biasShiftRotation_f6_cplus.restype = ctypes.c_double

        self.biasShiftRotation_f7_cplus = p.biasShiftRotation_f7_cplus
        self.biasShiftRotation_f7_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.biasShiftRotation_f7_cplus.restype = ctypes.c_double

        self.biasShiftRotation_f8_cplus = p.biasShiftRotation_f8_cplus
        self.biasShiftRotation_f8_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.biasShiftRotation_f8_cplus.restype = ctypes.c_double

        self.biasShiftRotation_f9_cplus = p.biasShiftRotation_f9_cplus
        self.biasShiftRotation_f9_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.biasShiftRotation_f9_cplus.restype = ctypes.c_double

        self.biasShiftRotation_f10_cplus = p.biasShiftRotation_f10_cplus
        self.biasShiftRotation_f10_cplus.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.biasShiftRotation_f10_cplus.restype = ctypes.c_double




class PortfolioOpt:
    def __init__(self, data_dir, market_name, dim, torch_flag=False):
        
        self.data_dir = data_dir # '.\data'
        self.market_name = market_name # 'SP500', 'CSI300, 'HSI', 'mix_SP500_CSI300_HSI'
        self.dim = dim
        self.lowest_sr = 1E-10
        
        if torch_flag:
            self.torch_flag = TORCH_FLAG
        else:
            self.torch_flag = False

        if self.market_name[:3] == 'mix':
            self.ismix = True
        else:
            self.ismix = False
        # https://www.investing.com/rates-bonds/u.s.-5-year-bond-yield-historical-data
        self.info_dict = {
            'CSI300': {
                'trading_days': 1213,
                'years': 5,
                'rf': 3.037, # risk-free rate
            },
            'SP500': {
                'trading_days': 1258,
                'years': 5,
                'rf': 1.6575,
            } ,
            'HSI': {
                'trading_days': 1233,
                'years': 5,
                'rf': 1.318, # risk-free rate
            } ,
            'DOW': {
                'trading_days': 1258,
                'years': 5,
                'rf': 1.6575, # risk-free rate
            } ,
            'mix_SP500_CSI300_HSI': {
                'trading_days': 1148,
                'years': 5,
                'rf': 2.0042, # average risk-free rate      
            }
        }

        self.trading_days = self.info_dict[self.market_name]['trading_days']
        self.years = self.info_dict[self.market_name]['years']
        self.rf = self.info_dict[self.market_name]['rf']
        if self.ismix:
            # Merge market trading       
            self.load_multi_mkt()     
        else:
            # Single market
            self.load_single_mkt()

        """
        Dim, market, merge?
        * column 'stock' indicate the rank of company in market index. (Some ranks may miss here as those companies do not have complete data during that period and have been filtered.)
        
        """
    def func(self, weights):
        weights = np.abs(weights)
        if np.sum(np.abs(weights)) == 0:
            print("All weights are zero..")
            sharpe_ratio = 0
        else:    
            weights = weights / np.sum(np.abs(weights))
            weights = np.matrix(weights)
            port_return = (np.sum(weights * self.avg_daily_return_matrix.T) * self.trading_days)/self.years # 1259 trading days over 5 year period
            
            # if self.torch_flag:
            #     weights_gpu = torch.from_numpy(weights)
            #     weights_gpu = weights_gpu.to(device)
            #     cov_gpu = torch.from_numpy(self.cov_matrix)
            #     cov_gpu = cov_gpu.to(device)
            #     std_gpu = float(torch.mm(torch.mm(weights_gpu, cov_gpu), weights_gpu.T))
            #     port_std_dev = np.sqrt(std_gpu) * np.sqrt(self.trading_days) / np.sqrt(self.years)
            # else:
            #     port_std_dev = (np.sqrt(weights * self.cov_matrix * weights.T) * np.sqrt(self.trading_days))/np.sqrt(self.years)
            
            port_std_dev = (np.sqrt(weights * self.cov_matrix * weights.T) * np.sqrt(self.trading_days))/np.sqrt(self.years)
            port_std_dev = float(port_std_dev)
            sharpe_ratio = (port_return - self.rf) / port_std_dev # 2.57 represents annual return of risk free security - 5-year US Treasury
        if sharpe_ratio <= 0:
            sharpe_ratio = self.lowest_sr
        
        fitness = 1.0/sharpe_ratio
        return fitness

    def final_eval(self, weights):
        weights = np.abs(weights)
        if np.sum(np.abs(weights)) == 0:
            print("All weights are zero..")
            port_return = 0
            port_std_dev = 0
            sharpe_ratio = 0
        else: 
            weights = weights / np.sum(np.abs(weights))
            weights = np.matrix(weights)
            port_return = (np.sum(weights * self.avg_daily_return_matrix.T) * self.trading_days)/self.years # 1259 trading days over 5 year period
            
            # if self.torch_flag:
            #     weights_gpu = torch.from_numpy(weights)
            #     weights_gpu = weights_gpu.to(device)
            #     cov_gpu = torch.from_numpy(self.cov_matrix)
            #     cov_gpu = cov_gpu.to(device)
            #     std_gpu = float(torch.mm(torch.mm(weights_gpu, cov_gpu), weights_gpu.T))
            #     port_std_dev = np.sqrt(std_gpu) * np.sqrt(self.trading_days) / np.sqrt(self.years)
            # else:
            #     port_std_dev = (np.sqrt(weights * self.cov_matrix * weights.T) * np.sqrt(self.trading_days))/np.sqrt(self.years)
        
            port_std_dev = (np.sqrt(weights * self.cov_matrix * weights.T) * np.sqrt(self.trading_days))/np.sqrt(self.years)
            port_std_dev = float(port_std_dev)
            sharpe_ratio = (port_return - self.rf) / port_std_dev # 2.57 represents annual return of risk free security - 5-year US Treasury
        
        if sharpe_ratio <= 0:
            sharpe_ratio = self.lowest_sr
        
        fitness = 1.0/sharpe_ratio
        return fitness, sharpe_ratio, port_return, port_std_dev

    def load_single_mkt(self):
        mkt_name = self.market_name
        fpath = os.path.join(self.data_dir, '{}.csv'.format(mkt_name))
        fdata = pd.read_csv(fpath, header=0)
        fdata['date'] = pd.to_datetime(fdata['date'])    
        fdata['daily_returns'] = fdata['daily_returns'] * 100 # actual value -> percentage value
        fdata.sort_values(['stock', 'date'], ascending=True, inplace=True)
        fdata.reset_index(drop=True, inplace=True)
        stock_lst = np.sort(fdata['stock'].unique(), axis=0)

        avg_daily_return_lst = []
        daily_return_lst = [] # for cov calculation 

        for idx, sig_stock in enumerate(stock_lst):
            if idx >= self.dim:
                break
            
            sig_data = copy.deepcopy(fdata[fdata['stock']==sig_stock])
            sig_data.sort_values('date', ascending=True, inplace=True)
            sig_data.reset_index(drop=True, inplace=True)
            avg_return = np.mean(sig_data['daily_returns']) # daily avg return
            avg_daily_return_lst.append(avg_return)
            daily_return_lst.append(list(sig_data['daily_returns']))

        self.avg_daily_return_matrix= np.matrix(avg_daily_return_lst)
        daily_return_lst = np.matrix(daily_return_lst)
        self.cov_matrix = np.cov(daily_return_lst)

        disp_str = "Market: {}, total stocks: {}, dim: {}".format(mkt_name, len(stock_lst), self.dim)
        print("-" * 30)
        print(disp_str)
        print('-' * 30)

    def load_multi_mkt(self):
        
        fpath = os.path.join(self.data_dir, "{}.csv".format(self.market_name))
        fdata = pd.read_csv(fpath, header=0)
        fdata['date'] = pd.to_datetime(fdata['date'])    
        fdata['daily_returns'] = fdata['daily_returns'] * 100 # actual value -> percentage value
        fdata.sort_values(['stock', 'date'], ascending=True, inplace=True)
        fdata.reset_index(drop=True, inplace=True)

        market_name_lst = self.market_name.split('_')[1:]
        num_of_market = len(market_name_lst)
        mkt_dict = {}

        for mkt_name in market_name_lst:
            mkt_dict[mkt_name] = {
                'num_of_stock': 0,
                'stock_list': [],
            }

        stock_lst = list(set(list(fdata['stock'])))
        total_num_stocks = len(stock_lst)
        disp_str = "Num of markets: {}, total stocks: {}, dim: {} \n".format(num_of_market, total_num_stocks, self.dim)
        
        for sig_stock in stock_lst:
            belong_mkt = sig_stock.split('-')[0]
            belong_rank = int(sig_stock.split('-')[1])
            mkt_dict[belong_mkt]['num_of_stock'] = mkt_dict[belong_mkt]['num_of_stock'] + 1
            mkt_dict[belong_mkt]['stock_list'].append(belong_rank)

        if self.dim == total_num_stocks:
            for mkt_idx, mkt_name in enumerate(market_name_lst):
                mkt_dict[mkt_name]['stock_list'] = np.sort(mkt_dict[mkt_name]['stock_list'], axis=0)
                mkt_dict[mkt_name]['select_num_stocks'] = mkt_dict[mkt_name]['num_of_stock']
                mkt_dict[mkt_name]['select_stocks'] = mkt_dict[mkt_name]['stock_list']
        else:
            acc_stock = 0
            for mkt_idx, mkt_name in enumerate(market_name_lst):
                if mkt_idx == (num_of_market - 1):
                    select_num = self.dim - acc_stock
                    if (select_num <= 0) or (select_num >  mkt_dict[mkt_name]['num_of_stock']):
                        raise ValueError("Unexpected selected number of stocks [{}], all number of stocks: [{}]..".format(select_num, mkt_dict[mkt_name]['num_of_stock']))
                else:
                    select_num = int(np.floor(mkt_dict[mkt_name]['num_of_stock'] * self.dim / total_num_stocks))
                    acc_stock = acc_stock + select_num
                mkt_dict[mkt_name]['stock_list'] = np.sort(mkt_dict[mkt_name]['stock_list'], axis=0)
                mkt_dict[mkt_name]['select_num_stocks'] = select_num
                mkt_dict[mkt_name]['select_stocks'] = mkt_dict[mkt_name]['stock_list'][:mkt_dict[mkt_name]['select_num_stocks']]

        avg_daily_return_lst = []
        daily_return_lst = [] # for cov calculation 
        for mkt_name in market_name_lst:
            disp_str = disp_str + "- Market: {}, selected stock: {}, total stocks: {} \n".format(mkt_name, mkt_dict[mkt_name]['select_num_stocks'], mkt_dict[mkt_name]['num_of_stock'])
            sig_mkt_stock_lst = mkt_dict[mkt_name]['select_stocks']
            for sig_stock in sig_mkt_stock_lst:
                stock_name = '{}-{}'.format(mkt_name, sig_stock)                    
                sig_data = copy.deepcopy(fdata[fdata['stock']==stock_name])
                sig_data.sort_values('date', ascending=True, inplace=True)
                sig_data.reset_index(drop=True, inplace=True)
                avg_return = np.mean(sig_data['daily_returns']) # daily avg return
                avg_daily_return_lst.append(avg_return)
                daily_return_lst.append(list(sig_data['daily_returns']))
                
        self.avg_daily_return_matrix= np.matrix(avg_daily_return_lst)
        daily_return_lst = np.matrix(daily_return_lst)
        self.cov_matrix = np.cov(daily_return_lst)
        
        print("-" * 30)
        print(disp_str)
        print('-' * 30)


# Degree of constraints
# c = the no. of times x variable appeared in the objective functions / the total no. of all vars appeared in the obejective function]
def varDegree(func_name, dim):
    
    var_degree_obj_func = {
        10: {
            "cec1-2014": [0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.11764705882352941, 0.11764705882352941, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706],
            "cec2-2014": [0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.11764705882352941],
            "cec3-2014": [0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.11764705882352941, 0.11764705882352941, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706],
            "cec4-2014": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "cec5-2014": [0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.11764705882352941, 0.11764705882352941, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706],
            "cec6-2014": [0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.11764705882352941],
            "cec7-2014": [0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.11764705882352941, 0.11764705882352941],
            "cec8-2014": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "cec9-2014": [0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.11764705882352941, 0.11764705882352941, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706],
            "cec10-2014": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "cec11-2014": [0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.11764705882352941, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941],
            "cec12-2014": [0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.11764705882352941, 0.11764705882352941, 0.08823529411764706],
            "cec13-2014": [0.11764705882352941, 0.11764705882352941, 0.08823529411764706, 0.11764705882352941, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706],
            "cec14-2014": [0.11764705882352941, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.11764705882352941],
            "cec15-2014": [0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.11764705882352941, 0.11764705882352941],
            "cec16-2014": [0.11764705882352941, 0.08823529411764706, 0.11764705882352941, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941],
            "cec17-2014": [0.09302325581395349, 0.09302325581395349, 0.13953488372093023, 0.09302325581395349, 0.06976744186046512, 0.13953488372093023, 0.06976744186046512, 0.06976744186046512, 0.09302325581395349, 0.13953488372093023],
            "cec18-2014": [0.1038961038961039, 0.15584415584415584, 0.15584415584415584, 0.03896103896103896, 0.15584415584415584, 0.1038961038961039, 0.03896103896103896, 0.1038961038961039, 0.03896103896103896, 0.1038961038961039],
            "cec19-2014": [0.041666666666666664, 0.125, 0.125, 0.125, 0.041666666666666664, 0.125, 0.125, 0.08333333333333333, 0.08333333333333333, 0.125],
            "cec20-2014": [0.03571428571428571, 0.10714285714285714, 0.10714285714285714, 0.10714285714285714, 0.03571428571428571, 0.10714285714285714, 0.10714285714285714, 0.10714285714285714, 0.14285714285714285, 0.14285714285714285],
            "cec21-2014": [0.05405405405405406, 0.08108108108108109, 0.05405405405405406, 0.21621621621621623, 0.05405405405405406, 0.08108108108108109, 0.08108108108108109, 0.08108108108108109, 0.21621621621621623, 0.08108108108108109],
            "cec22-2014": [0.09090909090909091, 0.13636363636363635, 0.045454545454545456, 0.13636363636363635, 0.09090909090909091, 0.13636363636363635, 0.045454545454545456, 0.13636363636363635, 0.13636363636363635, 0.045454545454545456],
            "cec23-2014": [0.10526315789473684, 0.09868421052631579, 0.09868421052631579, 0.09539473684210527, 0.1118421052631579, 0.09868421052631579, 0.10197368421052631, 0.08881578947368421, 0.09539473684210527, 0.10526315789473684],
            "cec24-2014": [0.10248447204968944, 0.11490683229813664, 0.11490683229813664, 0.07763975155279502, 0.09006211180124224, 0.10248447204968944, 0.07763975155279502, 0.11490683229813664, 0.11490683229813664, 0.09006211180124224],
            "cec25-2014": [0.10576923076923077, 0.11538461538461539, 0.10576923076923077, 0.08653846153846154, 0.08653846153846154, 0.10576923076923077, 0.11538461538461539, 0.08653846153846154, 0.10576923076923077, 0.08653846153846154],
            "cec26-2014": [0.10096153846153846, 0.09615384615384616, 0.09134615384615384, 0.10096153846153846, 0.11538461538461539, 0.11057692307692307, 0.09615384615384616, 0.09615384615384616, 0.10576923076923077, 0.08653846153846154],
            "cec27-2014": [0.08974358974358974, 0.1111111111111111, 0.0811965811965812, 0.11538461538461539, 0.08547008547008547, 0.11538461538461539, 0.10256410256410256, 0.10256410256410256, 0.08547008547008547, 0.1111111111111111],
            "cec28-2014": [0.1111111111111111, 0.09401709401709402, 0.08974358974358974, 0.09401709401709402, 0.11538461538461539, 0.08974358974358974, 0.09401709401709402, 0.09401709401709402, 0.1111111111111111, 0.10683760683760683],
            "cec29-2014": [0.0979020979020979, 0.12237762237762238, 0.1048951048951049, 0.0979020979020979, 0.08391608391608392, 0.10839160839160839, 0.09090909090909091, 0.06643356643356643, 0.1048951048951049, 0.12237762237762238],
            "cec30-2014": [0.07763975155279502, 0.10869565217391304, 0.12732919254658384, 0.12732919254658384, 0.07763975155279502, 0.07763975155279502, 0.10869565217391304, 0.12732919254658384, 0.07763975155279502, 0.09006211180124224],

        },
        20: {},
        30: {
            "cec1-2014": [0.010869565217391304, 0.04891304347826087, 0.010869565217391304, 0.03804347826086957, 0.04891304347826087, 0.03804347826086957, 0.03804347826086957, 0.02717391304347826, 0.02717391304347826, 0.04891304347826087, 0.016304347826086956, 0.016304347826086956, 0.021739130434782608, 0.04891304347826087, 0.03804347826086957, 0.02717391304347826, 0.02717391304347826, 0.021739130434782608, 0.04891304347826087, 0.016304347826086956, 0.03804347826086957, 0.04891304347826087, 0.04891304347826087, 0.03804347826086957, 0.021739130434782608, 0.021739130434782608, 0.03804347826086957, 0.02717391304347826, 0.04891304347826087, 0.04891304347826087],
            "cec2-2014": [0.010869565217391304, 0.04891304347826087, 0.02717391304347826, 0.04891304347826087, 0.021739130434782608, 0.03804347826086957, 0.02717391304347826, 0.04891304347826087, 0.016304347826086956, 0.03804347826086957, 0.04891304347826087, 0.02717391304347826, 0.021739130434782608, 0.04891304347826087, 0.03804347826086957, 0.016304347826086956, 0.02717391304347826, 0.021739130434782608, 0.04891304347826087, 0.04891304347826087, 0.021739130434782608, 0.03804347826086957, 0.010869565217391304, 0.016304347826086956, 0.04891304347826087, 0.04891304347826087, 0.03804347826086957, 0.02717391304347826, 0.03804347826086957, 0.03804347826086957],
            "cec3-2014": [0.016304347826086956, 0.021739130434782608, 0.03804347826086957, 0.010869565217391304, 0.02717391304347826, 0.021739130434782608, 0.04891304347826087, 0.03804347826086957, 0.04891304347826087, 0.04891304347826087, 0.02717391304347826, 0.04891304347826087, 0.04891304347826087, 0.02717391304347826, 0.016304347826086956, 0.04891304347826087, 0.021739130434782608, 0.03804347826086957, 0.04891304347826087, 0.04891304347826087, 0.02717391304347826, 0.04891304347826087, 0.010869565217391304, 0.03804347826086957, 0.021739130434782608, 0.02717391304347826, 0.03804347826086957, 0.03804347826086957, 0.03804347826086957, 0.016304347826086956],
            "cec4-2014": [0.03795066413662239, 0.022770398481973434, 0.04743833017077799, 0.04743833017077799, 0.028462998102466792, 0.028462998102466792, 0.022770398481973434, 0.03795066413662239, 0.04743833017077799, 0.04743833017077799, 0.011385199240986717, 0.028462998102466792, 0.03795066413662239, 0.04743833017077799, 0.04743833017077799, 0.03795066413662239, 0.04743833017077799, 0.017077798861480076, 0.028462998102466792, 0.011385199240986717, 0.03795066413662239, 0.017077798861480076, 0.022770398481973434, 0.04743833017077799, 0.03795066413662239, 0.022770398481973434, 0.017077798861480076, 0.03795066413662239, 0.028462998102466792, 0.04743833017077799],
            "cec5-2014": [0.04891304347826087, 0.016304347826086956, 0.04891304347826087, 0.03804347826086957, 0.04891304347826087, 0.03804347826086957, 0.02717391304347826, 0.04891304347826087, 0.04891304347826087, 0.04891304347826087, 0.03804347826086957, 0.021739130434782608, 0.03804347826086957, 0.021739130434782608, 0.04891304347826087, 0.03804347826086957, 0.010869565217391304, 0.02717391304347826, 0.04891304347826087, 0.02717391304347826, 0.04891304347826087, 0.010869565217391304, 0.021739130434782608, 0.021739130434782608, 0.016304347826086956, 0.016304347826086956, 0.02717391304347826, 0.02717391304347826, 0.03804347826086957, 0.03804347826086957],
            "cec6-2014": [0.02717391304347826, 0.02717391304347826, 0.010869565217391304, 0.04891304347826087, 0.03804347826086957, 0.03804347826086957, 0.04891304347826087, 0.04891304347826087, 0.04891304347826087, 0.02717391304347826, 0.021739130434782608, 0.03804347826086957, 0.010869565217391304, 0.04891304347826087, 0.03804347826086957, 0.021739130434782608, 0.021739130434782608, 0.04891304347826087, 0.04891304347826087, 0.016304347826086956, 0.04891304347826087, 0.016304347826086956, 0.021739130434782608, 0.04891304347826087, 0.016304347826086956, 0.03804347826086957, 0.02717391304347826, 0.03804347826086957, 0.03804347826086957, 0.02717391304347826],
            "cec7-2014": [0.021739130434782608, 0.016304347826086956, 0.04891304347826087, 0.04891304347826087, 0.021739130434782608, 0.021739130434782608, 0.02717391304347826, 0.02717391304347826, 0.02717391304347826, 0.04891304347826087, 0.04891304347826087, 0.03804347826086957, 0.04891304347826087, 0.010869565217391304, 0.04891304347826087, 0.016304347826086956, 0.03804347826086957, 0.03804347826086957, 0.03804347826086957, 0.010869565217391304, 0.021739130434782608, 0.03804347826086957, 0.016304347826086956, 0.02717391304347826, 0.03804347826086957, 0.04891304347826087, 0.04891304347826087, 0.02717391304347826, 0.03804347826086957, 0.04891304347826087],
            "cec8-2014": [0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333],
            "cec9-2014": [0.04891304347826087, 0.010869565217391304, 0.04891304347826087, 0.04891304347826087, 0.02717391304347826, 0.02717391304347826, 0.03804347826086957, 0.016304347826086956, 0.016304347826086956, 0.021739130434782608, 0.04891304347826087, 0.04891304347826087, 0.04891304347826087, 0.03804347826086957, 0.02717391304347826, 0.03804347826086957, 0.04891304347826087, 0.02717391304347826, 0.02717391304347826, 0.03804347826086957, 0.03804347826086957, 0.016304347826086956, 0.04891304347826087, 0.010869565217391304, 0.03804347826086957, 0.04891304347826087, 0.021739130434782608, 0.03804347826086957, 0.021739130434782608, 0.021739130434782608],
            "cec10-2014": [0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333],
            "cec11-2014": [0.03804347826086957, 0.016304347826086956, 0.04891304347826087, 0.04891304347826087, 0.02717391304347826, 0.021739130434782608, 0.010869565217391304, 0.021739130434782608, 0.016304347826086956, 0.02717391304347826, 0.04891304347826087, 0.04891304347826087, 0.03804347826086957, 0.021739130434782608, 0.03804347826086957, 0.04891304347826087, 0.03804347826086957, 0.03804347826086957, 0.02717391304347826, 0.02717391304347826, 0.010869565217391304, 0.04891304347826087, 0.016304347826086956, 0.04891304347826087, 0.03804347826086957, 0.04891304347826087, 0.04891304347826087, 0.02717391304347826, 0.03804347826086957, 0.021739130434782608],
            "cec12-2014": [0.03804347826086957, 0.02717391304347826, 0.04891304347826087, 0.02717391304347826, 0.016304347826086956, 0.04891304347826087, 0.03804347826086957, 0.02717391304347826, 0.010869565217391304, 0.03804347826086957, 0.04891304347826087, 0.03804347826086957, 0.03804347826086957, 0.04891304347826087, 0.010869565217391304, 0.04891304347826087, 0.021739130434782608, 0.04891304347826087, 0.02717391304347826, 0.021739130434782608, 0.04891304347826087, 0.021739130434782608, 0.03804347826086957, 0.04891304347826087, 0.03804347826086957, 0.04891304347826087, 0.021739130434782608, 0.016304347826086956, 0.016304347826086956, 0.02717391304347826],
            "cec13-2014": [0.03804347826086957, 0.021739130434782608, 0.04891304347826087, 0.016304347826086956, 0.021739130434782608, 0.016304347826086956, 0.03804347826086957, 0.04891304347826087, 0.03804347826086957, 0.03804347826086957, 0.04891304347826087, 0.016304347826086956, 0.04891304347826087, 0.021739130434782608, 0.02717391304347826, 0.021739130434782608, 0.02717391304347826, 0.04891304347826087, 0.02717391304347826, 0.02717391304347826, 0.04891304347826087, 0.04891304347826087, 0.03804347826086957, 0.04891304347826087, 0.010869565217391304, 0.04891304347826087, 0.03804347826086957, 0.03804347826086957, 0.010869565217391304, 0.02717391304347826],
            "cec14-2014": [0.021739130434782608, 0.02717391304347826, 0.016304347826086956, 0.03804347826086957, 0.03804347826086957, 0.03804347826086957, 0.016304347826086956, 0.04891304347826087, 0.04891304347826087, 0.02717391304347826, 0.02717391304347826, 0.021739130434782608, 0.021739130434782608, 0.03804347826086957, 0.04891304347826087, 0.04891304347826087, 0.04891304347826087, 0.03804347826086957, 0.03804347826086957, 0.010869565217391304, 0.04891304347826087, 0.010869565217391304, 0.03804347826086957, 0.016304347826086956, 0.04891304347826087, 0.04891304347826087, 0.021739130434782608, 0.02717391304347826, 0.04891304347826087, 0.02717391304347826],
            "cec15-2014": [0.03804347826086957, 0.04891304347826087, 0.021739130434782608, 0.016304347826086956, 0.03804347826086957, 0.016304347826086956, 0.02717391304347826, 0.010869565217391304, 0.02717391304347826, 0.010869565217391304, 0.021739130434782608, 0.03804347826086957, 0.03804347826086957, 0.04891304347826087, 0.02717391304347826, 0.02717391304347826, 0.04891304347826087, 0.02717391304347826, 0.04891304347826087, 0.04891304347826087, 0.021739130434782608, 0.016304347826086956, 0.04891304347826087, 0.03804347826086957, 0.04891304347826087, 0.021739130434782608, 0.03804347826086957, 0.04891304347826087, 0.04891304347826087, 0.03804347826086957],
            "cec16-2014": [0.02717391304347826, 0.03804347826086957, 0.04891304347826087, 0.03804347826086957, 0.016304347826086956, 0.02717391304347826, 0.04891304347826087, 0.04891304347826087, 0.03804347826086957, 0.021739130434782608, 0.010869565217391304, 0.02717391304347826, 0.02717391304347826, 0.021739130434782608, 0.03804347826086957, 0.04891304347826087, 0.016304347826086956, 0.021739130434782608, 0.021739130434782608, 0.04891304347826087, 0.04891304347826087, 0.04891304347826087, 0.04891304347826087, 0.03804347826086957, 0.010869565217391304, 0.016304347826086956, 0.04891304347826087, 0.03804347826086957, 0.02717391304347826, 0.03804347826086957],
            "cec17-2014": [0.023255813953488372, 0.023255813953488372, 0.031007751937984496, 0.023255813953488372, 0.031007751937984496, 0.046511627906976744, 0.031007751937984496, 0.046511627906976744, 0.031007751937984496, 0.023255813953488372, 0.046511627906976744, 0.031007751937984496, 0.046511627906976744, 0.023255813953488372, 0.046511627906976744, 0.031007751937984496, 0.031007751937984496, 0.031007751937984496, 0.023255813953488372, 0.023255813953488372, 0.023255813953488372, 0.046511627906976744, 0.031007751937984496, 0.046511627906976744, 0.046511627906976744, 0.031007751937984496, 0.023255813953488372, 0.046511627906976744, 0.031007751937984496, 0.031007751937984496],
            "cec18-2014": [0.03463203463203463, 0.03463203463203463, 0.03463203463203463, 0.05194805194805195, 0.05194805194805195, 0.012987012987012988, 0.012987012987012988, 0.05194805194805195, 0.03463203463203463, 0.05194805194805195, 0.05194805194805195, 0.03463203463203463, 0.03463203463203463, 0.03463203463203463, 0.03463203463203463, 0.03463203463203463, 0.012987012987012988, 0.05194805194805195, 0.012987012987012988, 0.03463203463203463, 0.05194805194805195, 0.012987012987012988, 0.012987012987012988, 0.03463203463203463, 0.05194805194805195, 0.012987012987012988, 0.012987012987012988, 0.012987012987012988, 0.03463203463203463, 0.05194805194805195],
            "cec19-2014": [0.04938271604938271, 0.037037037037037035, 0.024691358024691357, 0.04938271604938271, 0.037037037037037035, 0.037037037037037035, 0.024691358024691357, 0.04938271604938271, 0.024691358024691357, 0.04938271604938271, 0.012345679012345678, 0.012345679012345678, 0.04938271604938271, 0.04938271604938271, 0.024691358024691357, 0.012345679012345678, 0.037037037037037035, 0.037037037037037035, 0.04938271604938271, 0.037037037037037035, 0.04938271604938271, 0.024691358024691357, 0.012345679012345678, 0.012345679012345678, 0.037037037037037035, 0.037037037037037035, 0.04938271604938271, 0.024691358024691357, 0.012345679012345678, 0.037037037037037035],
            "cec20-2014": [0.047619047619047616, 0.011904761904761904, 0.03571428571428571, 0.047619047619047616, 0.047619047619047616, 0.03571428571428571, 0.03571428571428571, 0.03571428571428571, 0.03571428571428571, 0.03571428571428571, 0.03571428571428571, 0.047619047619047616, 0.03571428571428571, 0.03571428571428571, 0.011904761904761904, 0.03571428571428571, 0.03571428571428571, 0.03571428571428571, 0.011904761904761904, 0.011904761904761904, 0.011904761904761904, 0.03571428571428571, 0.047619047619047616, 0.047619047619047616, 0.03571428571428571, 0.03571428571428571, 0.03571428571428571, 0.03571428571428571, 0.03571428571428571, 0.011904761904761904],
            "cec21-2014": [0.024390243902439025, 0.04065040650406504, 0.06504065040650407, 0.024390243902439025, 0.04065040650406504, 0.04065040650406504, 0.024390243902439025, 0.024390243902439025, 0.06504065040650407, 0.016260162601626018, 0.024390243902439025, 0.016260162601626018, 0.016260162601626018, 0.024390243902439025, 0.024390243902439025, 0.016260162601626018, 0.04065040650406504, 0.016260162601626018, 0.06504065040650407, 0.06504065040650407, 0.06504065040650407, 0.016260162601626018, 0.04065040650406504, 0.016260162601626018, 0.016260162601626018, 0.06504065040650407, 0.04065040650406504, 0.024390243902439025, 0.016260162601626018, 0.024390243902439025],
            "cec22-2014": [0.015151515151515152, 0.045454545454545456, 0.015151515151515152, 0.015151515151515152, 0.030303030303030304, 0.045454545454545456, 0.030303030303030304, 0.045454545454545456, 0.015151515151515152, 0.015151515151515152, 0.045454545454545456, 0.015151515151515152, 0.015151515151515152, 0.045454545454545456, 0.045454545454545456, 0.045454545454545456, 0.030303030303030304, 0.045454545454545456, 0.045454545454545456, 0.045454545454545456, 0.015151515151515152, 0.045454545454545456, 0.045454545454545456, 0.045454545454545456, 0.015151515151515152, 0.045454545454545456, 0.030303030303030304, 0.045454545454545456, 0.030303030303030304, 0.030303030303030304],
            "cec23-2014": [0.03739837398373984, 0.03333333333333333, 0.034959349593495934, 0.036585365853658534, 0.03414634146341464, 0.03333333333333333, 0.030894308943089432, 0.034959349593495934, 0.03739837398373984, 0.03739837398373984, 0.032520325203252036, 0.028455284552845527, 0.03170731707317073, 0.02032520325203252, 0.03333333333333333, 0.03902439024390244, 0.034959349593495934, 0.03333333333333333, 0.03008130081300813, 0.034959349593495934, 0.03333333333333333, 0.03414634146341464, 0.036585365853658534, 0.034959349593495934, 0.036585365853658534, 0.01869918699186992, 0.03333333333333333, 0.03333333333333333, 0.03170731707317073, 0.038211382113821135],
            "cec24-2014": [0.019936204146730464, 0.03907496012759171, 0.03269537480063796, 0.03907496012759171, 0.019936204146730464, 0.03588516746411483, 0.029505582137161084, 0.03907496012759171, 0.03588516746411483, 0.03907496012759171, 0.03907496012759171, 0.029505582137161084, 0.023125996810207338, 0.03588516746411483, 0.03269537480063796, 0.029505582137161084, 0.029505582137161084, 0.029505582137161084, 0.029505582137161084, 0.03907496012759171, 0.03907496012759171, 0.03269537480063796, 0.03269537480063796, 0.03907496012759171, 0.03588516746411483, 0.03269537480063796, 0.03588516746411483, 0.03588516746411483, 0.029505582137161084, 0.03907496012759171],
            "cec25-2014": [0.031862745098039214, 0.031862745098039214, 0.03676470588235294, 0.022058823529411766, 0.03676470588235294, 0.02696078431372549, 0.03431372549019608, 0.03431372549019608, 0.03676470588235294, 0.0392156862745098, 0.0392156862745098, 0.031862745098039214, 0.03431372549019608, 0.0392156862745098, 0.0392156862745098, 0.024509803921568627, 0.031862745098039214, 0.03431372549019608, 0.03676470588235294, 0.029411764705882353, 0.03431372549019608, 0.031862745098039214, 0.03676470588235294, 0.024509803921568627, 0.031862745098039214, 0.029411764705882353, 0.031862745098039214, 0.031862745098039214, 0.0392156862745098, 0.03676470588235294],
            "cec26-2014": [0.03799019607843137, 0.03431372549019608, 0.030637254901960783, 0.03553921568627451, 0.03799019607843137, 0.03431372549019608, 0.03553921568627451, 0.03308823529411765, 0.01838235294117647, 0.03431372549019608, 0.029411764705882353, 0.03676470588235294, 0.03308823529411765, 0.03799019607843137, 0.030637254901960783, 0.031862745098039214, 0.031862745098039214, 0.03308823529411765, 0.03553921568627451, 0.03799019607843137, 0.03431372549019608, 0.03676470588235294, 0.0392156862745098, 0.023284313725490197, 0.023284313725490197, 0.03308823529411765, 0.03799019607843137, 0.030637254901960783, 0.03676470588235294, 0.03431372549019608],
            "cec27-2014": [0.03594771241830065, 0.030501089324618737, 0.03376906318082789, 0.034858387799564274, 0.030501089324618737, 0.0392156862745098, 0.03376906318082789, 0.02287581699346405, 0.03376906318082789, 0.032679738562091505, 0.030501089324618737, 0.03812636165577342, 0.03376906318082789, 0.030501089324618737, 0.034858387799564274, 0.037037037037037035, 0.032679738562091505, 0.02178649237472767, 0.03594771241830065, 0.034858387799564274, 0.0392156862745098, 0.03812636165577342, 0.029411764705882353, 0.037037037037037035, 0.027233115468409588, 0.03594771241830065, 0.030501089324618737, 0.034858387799564274, 0.03159041394335512, 0.03812636165577342],
            "cec28-2014": [0.03159041394335512, 0.0392156862745098, 0.03594771241830065, 0.027233115468409588, 0.02832244008714597, 0.03594771241830065, 0.034858387799564274, 0.030501089324618737, 0.030501089324618737, 0.032679738562091505, 0.029411764705882353, 0.03812636165577342, 0.03376906318082789, 0.0392156862745098, 0.03159041394335512, 0.026143790849673203, 0.034858387799564274, 0.03812636165577342, 0.03594771241830065, 0.03159041394335512, 0.037037037037037035, 0.03376906318082789, 0.026143790849673203, 0.03376906318082789, 0.03812636165577342, 0.037037037037037035, 0.03812636165577342, 0.026143790849673203, 0.03376906318082789, 0.030501089324618737],
            "cec29-2014": [0.05102040816326531, 0.03741496598639456, 0.030612244897959183, 0.036564625850340135, 0.04421768707482993, 0.031462585034013606, 0.03316326530612245, 0.025510204081632654, 0.022108843537414966, 0.03231292517006803, 0.02465986394557823, 0.030612244897959183, 0.02806122448979592, 0.023809523809523808, 0.02040816326530612, 0.03486394557823129, 0.031462585034013606, 0.030612244897959183, 0.04421768707482993, 0.034013605442176874, 0.050170068027210885, 0.0391156462585034, 0.013605442176870748, 0.03741496598639456, 0.034013605442176874, 0.04336734693877551, 0.03231292517006803, 0.026360544217687076, 0.031462585034013606, 0.045068027210884355],
            "cec30-2014": [0.030032467532467532, 0.026785714285714284, 0.0349025974025974, 0.03165584415584415, 0.026785714285714284, 0.020292207792207792, 0.030844155844155844, 0.03814935064935065, 0.032467532467532464, 0.030032467532467532, 0.03165584415584415, 0.03165584415584415, 0.03814935064935065, 0.03977272727272727, 0.02353896103896104, 0.030032467532467532, 0.036525974025974024, 0.04626623376623377, 0.041396103896103896, 0.03165584415584415, 0.0349025974025974, 0.0349025974025974, 0.041396103896103896, 0.041396103896103896, 0.03165584415584415, 0.03165584415584415, 0.0349025974025974, 0.0349025974025974, 0.03814935064935065, 0.02353896103896104],

        },
        50: {
            "cec1-2014": [0.02857142857142857, 0.02857142857142857, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.011428571428571429, 0.022857142857142857, 0.014285714285714285, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.011428571428571429, 0.017142857142857144, 0.008571428571428572, 0.017142857142857144, 0.02857142857142857, 0.014285714285714285, 0.022857142857142857, 0.017142857142857144, 0.02857142857142857, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.017142857142857144, 0.017142857142857144, 0.02857142857142857, 0.014285714285714285, 0.022857142857142857, 0.008571428571428572, 0.017142857142857144, 0.017142857142857144, 0.011428571428571429, 0.014285714285714285, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.011428571428571429, 0.008571428571428572, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144],
            "cec2-2014": [0.02857142857142857, 0.017142857142857144, 0.014285714285714285, 0.014285714285714285, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.017142857142857144, 0.011428571428571429, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.008571428571428572, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.011428571428571429, 0.017142857142857144, 0.017142857142857144, 0.02857142857142857, 0.008571428571428572, 0.02857142857142857, 0.02857142857142857, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.008571428571428572, 0.02857142857142857, 0.014285714285714285, 0.017142857142857144, 0.022857142857142857, 0.011428571428571429, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.011428571428571429, 0.017142857142857144, 0.017142857142857144],
            "cec3-2014": [0.022857142857142857, 0.011428571428571429, 0.017142857142857144, 0.02857142857142857, 0.008571428571428572, 0.02857142857142857, 0.02857142857142857, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.011428571428571429, 0.014285714285714285, 0.011428571428571429, 0.014285714285714285, 0.017142857142857144, 0.017142857142857144, 0.011428571428571429, 0.02857142857142857, 0.017142857142857144, 0.008571428571428572, 0.02857142857142857, 0.022857142857142857, 0.008571428571428572, 0.02857142857142857, 0.022857142857142857, 0.02857142857142857, 0.014285714285714285, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.014285714285714285, 0.017142857142857144, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.02857142857142857, 0.014285714285714285],
            "cec4-2014": [0.016536964980544747, 0.021400778210116732, 0.014591439688715954, 0.021400778210116732, 0.023346303501945526, 0.014591439688715954, 0.029182879377431907, 0.008754863813229572, 0.021400778210116732, 0.029182879377431907, 0.023346303501945526, 0.011673151750972763, 0.016536964980544747, 0.017509727626459144, 0.029182879377431907, 0.023346303501945526, 0.023346303501945526, 0.017509727626459144, 0.023346303501945526, 0.014591439688715954, 0.016536964980544747, 0.008754863813229572, 0.014591439688715954, 0.017509727626459144, 0.029182879377431907, 0.021400778210116732, 0.029182879377431907, 0.029182879377431907, 0.017509727626459144, 0.023346303501945526, 0.011673151750972763, 0.023346303501945526, 0.008754863813229572, 0.017509727626459144, 0.011673151750972763, 0.021400778210116732, 0.021400778210116732, 0.016536964980544747, 0.029182879377431907, 0.021400778210116732, 0.014591439688715954, 0.023346303501945526, 0.011673151750972763, 0.016536964980544747, 0.029182879377431907, 0.016536964980544747, 0.017509727626459144, 0.029182879377431907, 0.029182879377431907, 0.021400778210116732],
            "cec5-2014": [0.02857142857142857, 0.011428571428571429, 0.017142857142857144, 0.008571428571428572, 0.017142857142857144, 0.017142857142857144, 0.02857142857142857, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.017142857142857144, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.011428571428571429, 0.017142857142857144, 0.011428571428571429, 0.02857142857142857, 0.022857142857142857, 0.02857142857142857, 0.02857142857142857, 0.011428571428571429, 0.017142857142857144, 0.02857142857142857, 0.017142857142857144, 0.017142857142857144, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.017142857142857144, 0.022857142857142857, 0.02857142857142857, 0.008571428571428572, 0.022857142857142857, 0.008571428571428572, 0.014285714285714285, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.022857142857142857, 0.017142857142857144],
            "cec6-2014": [0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.02857142857142857, 0.022857142857142857, 0.014285714285714285, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.02857142857142857, 0.022857142857142857, 0.011428571428571429, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.014285714285714285, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.011428571428571429, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.017142857142857144, 0.011428571428571429, 0.02857142857142857, 0.022857142857142857, 0.014285714285714285, 0.011428571428571429, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.008571428571428572, 0.02857142857142857, 0.02857142857142857, 0.017142857142857144, 0.017142857142857144, 0.008571428571428572, 0.017142857142857144, 0.02857142857142857, 0.008571428571428572, 0.017142857142857144, 0.017142857142857144, 0.014285714285714285, 0.022857142857142857, 0.017142857142857144],
            "cec7-2014": [0.02857142857142857, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.02857142857142857, 0.008571428571428572, 0.017142857142857144, 0.008571428571428572, 0.022857142857142857, 0.017142857142857144, 0.02857142857142857, 0.017142857142857144, 0.014285714285714285, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.014285714285714285, 0.011428571428571429, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.011428571428571429, 0.022857142857142857, 0.014285714285714285, 0.011428571428571429, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.017142857142857144, 0.02857142857142857, 0.017142857142857144, 0.02857142857142857, 0.014285714285714285, 0.017142857142857144, 0.022857142857142857, 0.011428571428571429, 0.008571428571428572, 0.017142857142857144, 0.017142857142857144, 0.02857142857142857, 0.017142857142857144, 0.02857142857142857],
            "cec8-2014": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            "cec9-2014": [0.017142857142857144, 0.02857142857142857, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.02857142857142857, 0.02857142857142857, 0.014285714285714285, 0.022857142857142857, 0.011428571428571429, 0.022857142857142857, 0.017142857142857144, 0.017142857142857144, 0.011428571428571429, 0.008571428571428572, 0.008571428571428572, 0.02857142857142857, 0.014285714285714285, 0.022857142857142857, 0.014285714285714285, 0.02857142857142857, 0.022857142857142857, 0.014285714285714285, 0.02857142857142857, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.011428571428571429, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.014285714285714285, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.008571428571428572, 0.011428571428571429, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.017142857142857144, 0.017142857142857144, 0.02857142857142857],
            "cec10-2014": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            "cec11-2014": [0.022857142857142857, 0.017142857142857144, 0.008571428571428572, 0.022857142857142857, 0.022857142857142857, 0.011428571428571429, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.02857142857142857, 0.02857142857142857, 0.008571428571428572, 0.008571428571428572, 0.017142857142857144, 0.017142857142857144, 0.017142857142857144, 0.017142857142857144, 0.017142857142857144, 0.017142857142857144, 0.014285714285714285, 0.02857142857142857, 0.014285714285714285, 0.022857142857142857, 0.011428571428571429, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.014285714285714285, 0.02857142857142857, 0.022857142857142857, 0.017142857142857144, 0.02857142857142857, 0.011428571428571429, 0.022857142857142857, 0.014285714285714285, 0.02857142857142857, 0.011428571428571429, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.02857142857142857, 0.022857142857142857],
            "cec12-2014": [0.017142857142857144, 0.011428571428571429, 0.008571428571428572, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.02857142857142857, 0.014285714285714285, 0.02857142857142857, 0.008571428571428572, 0.014285714285714285, 0.022857142857142857, 0.017142857142857144, 0.014285714285714285, 0.022857142857142857, 0.022857142857142857, 0.011428571428571429, 0.011428571428571429, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.008571428571428572, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.02857142857142857, 0.02857142857142857, 0.017142857142857144, 0.02857142857142857, 0.022857142857142857, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.02857142857142857, 0.017142857142857144, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.011428571428571429, 0.017142857142857144, 0.014285714285714285, 0.017142857142857144],
            "cec13-2014": [0.022857142857142857, 0.011428571428571429, 0.014285714285714285, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.014285714285714285, 0.022857142857142857, 0.017142857142857144, 0.02857142857142857, 0.014285714285714285, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.011428571428571429, 0.011428571428571429, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.022857142857142857, 0.022857142857142857, 0.011428571428571429, 0.008571428571428572, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.017142857142857144, 0.02857142857142857, 0.02857142857142857, 0.017142857142857144, 0.02857142857142857, 0.02857142857142857, 0.02857142857142857, 0.02857142857142857, 0.02857142857142857, 0.022857142857142857, 0.014285714285714285, 0.008571428571428572, 0.008571428571428572, 0.017142857142857144],
            "cec14-2014": [0.011428571428571429, 0.022857142857142857, 0.011428571428571429, 0.022857142857142857, 0.017142857142857144, 0.008571428571428572, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.017142857142857144, 0.02857142857142857, 0.022857142857142857, 0.017142857142857144, 0.014285714285714285, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.014285714285714285, 0.02857142857142857, 0.011428571428571429, 0.017142857142857144, 0.022857142857142857, 0.008571428571428572, 0.022857142857142857, 0.017142857142857144, 0.02857142857142857, 0.02857142857142857, 0.014285714285714285, 0.02857142857142857, 0.011428571428571429, 0.017142857142857144, 0.017142857142857144, 0.008571428571428572, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.02857142857142857, 0.014285714285714285, 0.022857142857142857, 0.014285714285714285, 0.02857142857142857],
            "cec15-2014": [0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.017142857142857144, 0.02857142857142857, 0.022857142857142857, 0.008571428571428572, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.014285714285714285, 0.017142857142857144, 0.022857142857142857, 0.011428571428571429, 0.02857142857142857, 0.017142857142857144, 0.022857142857142857, 0.02857142857142857, 0.014285714285714285, 0.011428571428571429, 0.02857142857142857, 0.017142857142857144, 0.014285714285714285, 0.014285714285714285, 0.017142857142857144, 0.02857142857142857, 0.011428571428571429, 0.02857142857142857, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.017142857142857144, 0.014285714285714285, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.008571428571428572, 0.011428571428571429, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.008571428571428572, 0.022857142857142857, 0.022857142857142857],
            "cec16-2014": [0.017142857142857144, 0.017142857142857144, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.011428571428571429, 0.008571428571428572, 0.011428571428571429, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.02857142857142857, 0.02857142857142857, 0.017142857142857144, 0.022857142857142857, 0.02857142857142857, 0.02857142857142857, 0.02857142857142857, 0.017142857142857144, 0.017142857142857144, 0.011428571428571429, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.022857142857142857, 0.008571428571428572, 0.02857142857142857, 0.014285714285714285, 0.017142857142857144, 0.022857142857142857, 0.02857142857142857, 0.014285714285714285, 0.014285714285714285, 0.022857142857142857, 0.008571428571428572, 0.014285714285714285, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.011428571428571429, 0.02857142857142857, 0.022857142857142857],
            "cec17-2014": [0.013953488372093023, 0.013953488372093023, 0.018604651162790697, 0.018604651162790697, 0.027906976744186046, 0.027906976744186046, 0.027906976744186046, 0.018604651162790697, 0.027906976744186046, 0.013953488372093023, 0.027906976744186046, 0.018604651162790697, 0.027906976744186046, 0.027906976744186046, 0.013953488372093023, 0.027906976744186046, 0.018604651162790697, 0.013953488372093023, 0.013953488372093023, 0.013953488372093023, 0.027906976744186046, 0.018604651162790697, 0.018604651162790697, 0.027906976744186046, 0.013953488372093023, 0.018604651162790697, 0.013953488372093023, 0.018604651162790697, 0.013953488372093023, 0.018604651162790697, 0.018604651162790697, 0.018604651162790697, 0.018604651162790697, 0.013953488372093023, 0.013953488372093023, 0.027906976744186046, 0.018604651162790697, 0.018604651162790697, 0.013953488372093023, 0.027906976744186046, 0.018604651162790697, 0.027906976744186046, 0.018604651162790697, 0.027906976744186046, 0.018604651162790697, 0.018604651162790697, 0.013953488372093023, 0.027906976744186046, 0.018604651162790697, 0.013953488372093023],
            "cec18-2014": [0.02077922077922078, 0.03116883116883117, 0.007792207792207792, 0.02077922077922078, 0.03116883116883117, 0.02077922077922078, 0.02077922077922078, 0.02077922077922078, 0.007792207792207792, 0.007792207792207792, 0.007792207792207792, 0.02077922077922078, 0.02077922077922078, 0.007792207792207792, 0.007792207792207792, 0.02077922077922078, 0.03116883116883117, 0.02077922077922078, 0.007792207792207792, 0.03116883116883117, 0.007792207792207792, 0.02077922077922078, 0.007792207792207792, 0.03116883116883117, 0.03116883116883117, 0.02077922077922078, 0.02077922077922078, 0.03116883116883117, 0.03116883116883117, 0.03116883116883117, 0.007792207792207792, 0.007792207792207792, 0.03116883116883117, 0.02077922077922078, 0.007792207792207792, 0.03116883116883117, 0.007792207792207792, 0.03116883116883117, 0.007792207792207792, 0.03116883116883117, 0.02077922077922078, 0.03116883116883117, 0.02077922077922078, 0.02077922077922078, 0.02077922077922078, 0.02077922077922078, 0.007792207792207792, 0.02077922077922078, 0.03116883116883117, 0.02077922077922078],
            "cec19-2014": [0.021739130434782608, 0.007246376811594203, 0.007246376811594203, 0.007246376811594203, 0.021739130434782608, 0.021739130434782608, 0.007246376811594203, 0.014492753623188406, 0.030434782608695653, 0.021739130434782608, 0.021739130434782608, 0.021739130434782608, 0.014492753623188406, 0.007246376811594203, 0.030434782608695653, 0.030434782608695653, 0.030434782608695653, 0.007246376811594203, 0.007246376811594203, 0.014492753623188406, 0.021739130434782608, 0.021739130434782608, 0.007246376811594203, 0.030434782608695653, 0.030434782608695653, 0.030434782608695653, 0.030434782608695653, 0.014492753623188406, 0.030434782608695653, 0.021739130434782608, 0.014492753623188406, 0.007246376811594203, 0.021739130434782608, 0.014492753623188406, 0.030434782608695653, 0.021739130434782608, 0.021739130434782608, 0.030434782608695653, 0.021739130434782608, 0.014492753623188406, 0.030434782608695653, 0.014492753623188406, 0.030434782608695653, 0.014492753623188406, 0.030434782608695653, 0.030434782608695653, 0.021739130434782608, 0.021739130434782608, 0.014492753623188406, 0.007246376811594203],
            "cec20-2014": [0.007142857142857143, 0.02857142857142857, 0.02142857142857143, 0.02142857142857143, 0.02857142857142857, 0.02142857142857143, 0.02142857142857143, 0.02142857142857143, 0.02142857142857143, 0.02142857142857143, 0.02142857142857143, 0.007142857142857143, 0.007142857142857143, 0.007142857142857143, 0.007142857142857143, 0.02142857142857143, 0.02142857142857143, 0.02857142857142857, 0.02142857142857143, 0.02142857142857143, 0.007142857142857143, 0.02142857142857143, 0.02142857142857143, 0.02142857142857143, 0.02142857142857143, 0.02142857142857143, 0.007142857142857143, 0.02142857142857143, 0.02142857142857143, 0.02857142857142857, 0.02142857142857143, 0.02857142857142857, 0.02142857142857143, 0.007142857142857143, 0.02142857142857143, 0.02142857142857143, 0.02142857142857143, 0.02142857142857143, 0.007142857142857143, 0.02857142857142857, 0.02857142857142857, 0.02142857142857143, 0.02857142857142857, 0.02142857142857143, 0.02857142857142857, 0.007142857142857143, 0.02142857142857143, 0.02857142857142857, 0.02142857142857143, 0.02142857142857143],
            "cec21-2014": [0.02583732057416268, 0.009569377990430622, 0.014354066985645933, 0.009569377990430622, 0.03827751196172249, 0.02583732057416268, 0.009569377990430622, 0.014354066985645933, 0.014354066985645933, 0.014354066985645933, 0.009569377990430622, 0.014354066985645933, 0.009569377990430622, 0.009569377990430622, 0.03827751196172249, 0.014354066985645933, 0.009569377990430622, 0.014354066985645933, 0.02583732057416268, 0.009569377990430622, 0.014354066985645933, 0.009569377990430622, 0.014354066985645933, 0.014354066985645933, 0.014354066985645933, 0.009569377990430622, 0.009569377990430622, 0.02583732057416268, 0.02583732057416268, 0.03827751196172249, 0.009569377990430622, 0.03827751196172249, 0.009569377990430622, 0.014354066985645933, 0.03827751196172249, 0.03827751196172249, 0.014354066985645933, 0.03827751196172249, 0.03827751196172249, 0.02583732057416268, 0.009569377990430622, 0.02583732057416268, 0.03827751196172249, 0.014354066985645933, 0.02583732057416268, 0.014354066985645933, 0.03827751196172249, 0.009569377990430622, 0.02583732057416268, 0.02583732057416268],
            "cec22-2014": [0.02727272727272727, 0.00909090909090909, 0.00909090909090909, 0.00909090909090909, 0.02727272727272727, 0.02727272727272727, 0.00909090909090909, 0.02727272727272727, 0.02727272727272727, 0.01818181818181818, 0.02727272727272727, 0.02727272727272727, 0.02727272727272727, 0.02727272727272727, 0.02727272727272727, 0.02727272727272727, 0.02727272727272727, 0.01818181818181818, 0.00909090909090909, 0.01818181818181818, 0.00909090909090909, 0.02727272727272727, 0.00909090909090909, 0.00909090909090909, 0.01818181818181818, 0.02727272727272727, 0.02727272727272727, 0.00909090909090909, 0.02727272727272727, 0.01818181818181818, 0.00909090909090909, 0.01818181818181818, 0.00909090909090909, 0.02727272727272727, 0.02727272727272727, 0.00909090909090909, 0.02727272727272727, 0.01818181818181818, 0.02727272727272727, 0.00909090909090909, 0.02727272727272727, 0.00909090909090909, 0.02727272727272727, 0.02727272727272727, 0.01818181818181818, 0.00909090909090909, 0.02727272727272727, 0.01818181818181818, 0.01818181818181818, 0.02727272727272727],
            "cec23-2014": [0.018806214227309895, 0.019215044971381847, 0.019215044971381847, 0.022485690923957483, 0.02330335241210139, 0.019623875715453803, 0.016762060506950123, 0.020032706459525755, 0.020032706459525755, 0.0241210139002453, 0.020032706459525755, 0.01839738348323794, 0.020032706459525755, 0.008585445625511038, 0.02044153720359771, 0.016762060506950123, 0.0241210139002453, 0.01839738348323794, 0.019215044971381847, 0.022485690923957483, 0.02044153720359771, 0.022485690923957483, 0.0241210139002453, 0.020032706459525755, 0.01839738348323794, 0.022485690923957483, 0.02330335241210139, 0.019215044971381847, 0.02330335241210139, 0.015944399018806215, 0.019215044971381847, 0.021668029435813575, 0.01839738348323794, 0.019215044971381847, 0.011038430089942763, 0.020850367947669663, 0.018806214227309895, 0.022485690923957483, 0.019623875715453803, 0.019215044971381847, 0.022485690923957483, 0.020850367947669663, 0.02044153720359771, 0.01839738348323794, 0.020032706459525755, 0.022076860179885527, 0.020032706459525755, 0.020032706459525755, 0.022485690923957483, 0.020850367947669663],
            "cec24-2014": [0.02465642683912692, 0.016572352465642683, 0.019805982215036377, 0.02465642683912692, 0.010105092966855295, 0.02465642683912692, 0.019805982215036377, 0.021422797089733225, 0.02465642683912692, 0.014955537590945837, 0.02465642683912692, 0.019805982215036377, 0.014955537590945837, 0.023039611964430072, 0.02465642683912692, 0.023039611964430072, 0.01818916734033953, 0.014955537590945837, 0.021422797089733225, 0.021422797089733225, 0.019805982215036377, 0.01818916734033953, 0.01818916734033953, 0.014955537590945837, 0.01818916734033953, 0.021422797089733225, 0.02465642683912692, 0.011721907841552142, 0.02465642683912692, 0.019805982215036377, 0.02465642683912692, 0.01818916734033953, 0.01818916734033953, 0.01818916734033953, 0.023039611964430072, 0.023039611964430072, 0.02465642683912692, 0.023039611964430072, 0.021422797089733225, 0.01818916734033953, 0.016572352465642683, 0.01818916734033953, 0.021422797089733225, 0.011721907841552142, 0.021422797089733225, 0.01818916734033953, 0.023039611964430072, 0.019805982215036377, 0.016572352465642683, 0.021422797089733225],
            "cec25-2014": [0.023514851485148515, 0.018564356435643563, 0.019801980198019802, 0.019801980198019802, 0.018564356435643563, 0.019801980198019802, 0.017326732673267328, 0.01485148514851485, 0.01485148514851485, 0.01485148514851485, 0.022277227722772276, 0.023514851485148515, 0.019801980198019802, 0.023514851485148515, 0.023514851485148515, 0.024752475247524754, 0.01485148514851485, 0.017326732673267328, 0.019801980198019802, 0.022277227722772276, 0.02103960396039604, 0.017326732673267328, 0.02103960396039604, 0.01485148514851485, 0.022277227722772276, 0.022277227722772276, 0.018564356435643563, 0.019801980198019802, 0.017326732673267328, 0.02103960396039604, 0.018564356435643563, 0.02103960396039604, 0.02103960396039604, 0.019801980198019802, 0.019801980198019802, 0.023514851485148515, 0.022277227722772276, 0.018564356435643563, 0.019801980198019802, 0.02103960396039604, 0.017326732673267328, 0.022277227722772276, 0.012376237623762377, 0.024752475247524754, 0.018564356435643563, 0.022277227722772276, 0.023514851485148515, 0.024752475247524754, 0.018564356435643563, 0.02103960396039604],
            "cec26-2014": [0.02042079207920792, 0.022896039603960396, 0.019801980198019802, 0.01608910891089109, 0.01608910891089109, 0.023514851485148515, 0.017326732673267328, 0.024133663366336634, 0.02042079207920792, 0.019183168316831683, 0.017326732673267328, 0.019801980198019802, 0.021658415841584157, 0.021658415841584157, 0.022277227722772276, 0.023514851485148515, 0.022896039603960396, 0.022277227722772276, 0.019801980198019802, 0.021658415841584157, 0.02042079207920792, 0.019183168316831683, 0.02042079207920792, 0.02042079207920792, 0.01670792079207921, 0.02103960396039604, 0.02103960396039604, 0.023514851485148515, 0.018564356435643563, 0.01547029702970297, 0.023514851485148515, 0.019183168316831683, 0.02042079207920792, 0.018564356435643563, 0.017945544554455444, 0.01608910891089109, 0.02103960396039604, 0.017945544554455444, 0.02103960396039604, 0.019183168316831683, 0.02042079207920792, 0.023514851485148515, 0.021658415841584157, 0.02103960396039604, 0.021658415841584157, 0.014232673267326733, 0.02103960396039604, 0.012995049504950494, 0.023514851485148515, 0.01547029702970297],
            "cec27-2014": [0.020902090209020903, 0.022002200220022004, 0.019801980198019802, 0.017051705170517052, 0.02145214521452145, 0.020902090209020903, 0.019251925192519254, 0.0187018701870187, 0.0176017601760176, 0.02145214521452145, 0.022002200220022004, 0.0165016501650165, 0.0165016501650165, 0.019801980198019802, 0.0242024202420242, 0.02145214521452145, 0.02035203520352035, 0.018151815181518153, 0.022002200220022004, 0.018151815181518153, 0.0231023102310231, 0.0176017601760176, 0.023652365236523653, 0.019251925192519254, 0.018151815181518153, 0.014301430143014302, 0.0231023102310231, 0.022552255225522552, 0.023652365236523653, 0.02035203520352035, 0.019251925192519254, 0.019251925192519254, 0.02145214521452145, 0.018151815181518153, 0.0176017601760176, 0.018151815181518153, 0.0187018701870187, 0.020902090209020903, 0.013751375137513752, 0.02145214521452145, 0.02145214521452145, 0.0242024202420242, 0.018151815181518153, 0.022002200220022004, 0.0176017601760176, 0.0242024202420242, 0.02145214521452145, 0.022552255225522552, 0.0165016501650165, 0.019251925192519254],
            "cec28-2014": [0.020902090209020903, 0.019801980198019802, 0.015401540154015401, 0.017051705170517052, 0.02035203520352035, 0.02035203520352035, 0.022002200220022004, 0.022552255225522552, 0.024752475247524754, 0.022552255225522552, 0.0187018701870187, 0.018151815181518153, 0.02145214521452145, 0.0231023102310231, 0.01595159515951595, 0.022552255225522552, 0.014301430143014302, 0.022552255225522552, 0.02035203520352035, 0.0176017601760176, 0.02035203520352035, 0.02035203520352035, 0.0187018701870187, 0.0176017601760176, 0.019801980198019802, 0.019801980198019802, 0.02035203520352035, 0.02035203520352035, 0.020902090209020903, 0.018151815181518153, 0.019251925192519254, 0.019801980198019802, 0.02145214521452145, 0.02035203520352035, 0.019801980198019802, 0.022552255225522552, 0.02145214521452145, 0.019251925192519254, 0.02145214521452145, 0.02145214521452145, 0.01595159515951595, 0.0165016501650165, 0.020902090209020903, 0.0242024202420242, 0.019251925192519254, 0.022002200220022004, 0.019251925192519254, 0.0165016501650165, 0.02145214521452145, 0.02035203520352035],
            "cec29-2014": [0.020206362854686157, 0.026225279449699053, 0.011607910576096303, 0.018056749785038694, 0.01633705932932072, 0.024505588993981083, 0.019776440240756664, 0.021926053310404127, 0.012897678417884782, 0.018486672398968184, 0.026225279449699053, 0.021496130696474634, 0.024935511607910577, 0.013327601031814273, 0.008598452278589854, 0.013757523645743766, 0.026655202063628546, 0.018056749785038694, 0.01633705932932072, 0.010318142734307825, 0.027944969905417026, 0.025795356835769563, 0.026225279449699053, 0.023215821152192607, 0.022785898538263114, 0.01117798796216681, 0.02536543422184007, 0.01117798796216681, 0.014617368873602751, 0.021496130696474634, 0.014617368873602751, 0.026225279449699053, 0.021496130696474634, 0.01633705932932072, 0.027944969905417026, 0.02880481513327601, 0.02536543422184007, 0.012897678417884782, 0.023215821152192607, 0.016766981943250214, 0.012897678417884782, 0.027515047291487533, 0.024505588993981083, 0.01934651762682717, 0.01934651762682717, 0.022785898538263114, 0.02536543422184007, 0.02063628546861565, 0.013757523645743766, 0.02063628546861565],
            "cec30-2014": [0.019583333333333335, 0.015416666666666667, 0.014583333333333334, 0.020416666666666666, 0.014583333333333334, 0.016666666666666666, 0.020833333333333332, 0.009166666666666667, 0.017916666666666668, 0.020416666666666666, 0.019583333333333335, 0.02375, 0.014583333333333334, 0.020416666666666666, 0.02666666666666667, 0.009583333333333333, 0.02375, 0.0275, 0.019583333333333335, 0.019583333333333335, 0.01875, 0.024583333333333332, 0.02625, 0.017083333333333332, 0.020416666666666666, 0.020416666666666666, 0.025, 0.02, 0.020833333333333332, 0.028333333333333332, 0.020833333333333332, 0.025416666666666667, 0.02666666666666667, 0.01625, 0.02, 0.01625, 0.015416666666666667, 0.024166666666666666, 0.02, 0.019583333333333335, 0.020833333333333332, 0.022916666666666665, 0.02, 0.014583333333333334, 0.022083333333333333, 0.01625, 0.019583333333333335, 0.02375, 0.018333333333333333, 0.020833333333333332],            
        },
        100: {
            "cec1-2014": [0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.005555555555555556, 0.007407407407407408, 0.007407407407407408, 0.007407407407407408, 0.012962962962962963, 0.007407407407407408, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.009259259259259259, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.005555555555555556, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.005555555555555556, 0.011111111111111112, 0.005555555555555556, 0.007407407407407408, 0.007407407407407408, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.009259259259259259, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.005555555555555556, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.011111111111111112, 0.011111111111111112],
            "cec2-2014": [0.009259259259259259, 0.005555555555555556, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.005555555555555556, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.005555555555555556, 0.007407407407407408, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.007407407407407408, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.005555555555555556, 0.005555555555555556, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.007407407407407408, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.005555555555555556, 0.011111111111111112, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112],
            "cec3-2014": [0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.007407407407407408, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.007407407407407408, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.005555555555555556, 0.012962962962962963, 0.007407407407407408, 0.009259259259259259, 0.011111111111111112, 0.007407407407407408, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.005555555555555556, 0.007407407407407408, 0.007407407407407408, 0.009259259259259259, 0.011111111111111112, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.005555555555555556, 0.007407407407407408],
            "cec4-2014": [0.010299625468164793, 0.00749063670411985, 0.00749063670411985, 0.011235955056179775, 0.013108614232209739, 0.010299625468164793, 0.00749063670411985, 0.0056179775280898875, 0.009363295880149813, 0.009363295880149813, 0.009363295880149813, 0.013108614232209739, 0.013108614232209739, 0.011235955056179775, 0.0056179775280898875, 0.0056179775280898875, 0.009363295880149813, 0.0056179775280898875, 0.011235955056179775, 0.013108614232209739, 0.010299625468164793, 0.013108614232209739, 0.013108614232209739, 0.010299625468164793, 0.010299625468164793, 0.00749063670411985, 0.010299625468164793, 0.0056179775280898875, 0.009363295880149813, 0.00749063670411985, 0.009363295880149813, 0.009363295880149813, 0.009363295880149813, 0.011235955056179775, 0.013108614232209739, 0.013108614232209739, 0.013108614232209739, 0.00749063670411985, 0.011235955056179775, 0.0056179775280898875, 0.00749063670411985, 0.010299625468164793, 0.010299625468164793, 0.00749063670411985, 0.011235955056179775, 0.013108614232209739, 0.010299625468164793, 0.013108614232209739, 0.00749063670411985, 0.009363295880149813, 0.00749063670411985, 0.00749063670411985, 0.013108614232209739, 0.011235955056179775, 0.010299625468164793, 0.011235955056179775, 0.013108614232209739, 0.00749063670411985, 0.011235955056179775, 0.013108614232209739, 0.0056179775280898875, 0.013108614232209739, 0.013108614232209739, 0.009363295880149813, 0.013108614232209739, 0.00749063670411985, 0.011235955056179775, 0.0056179775280898875, 0.009363295880149813, 0.0056179775280898875, 0.009363295880149813, 0.013108614232209739, 0.009363295880149813, 0.00749063670411985, 0.009363295880149813, 0.013108614232209739, 0.00749063670411985, 0.013108614232209739, 0.011235955056179775, 0.013108614232209739, 0.011235955056179775, 0.0056179775280898875, 0.0056179775280898875, 0.009363295880149813, 0.0056179775280898875, 0.009363295880149813, 0.013108614232209739, 0.010299625468164793, 0.013108614232209739, 0.009363295880149813, 0.009363295880149813, 0.013108614232209739, 0.013108614232209739, 0.009363295880149813, 0.00749063670411985, 0.013108614232209739, 0.009363295880149813, 0.013108614232209739, 0.013108614232209739, 0.010299625468164793],
            "cec5-2014": [0.011111111111111112, 0.009259259259259259, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.012962962962962963, 0.005555555555555556, 0.005555555555555556, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.012962962962962963, 0.005555555555555556, 0.007407407407407408, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.011111111111111112, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.005555555555555556, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.005555555555555556, 0.007407407407407408, 0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.005555555555555556, 0.011111111111111112, 0.007407407407407408, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963],
            "cec6-2014": [0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.005555555555555556, 0.005555555555555556, 0.011111111111111112, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.005555555555555556, 0.007407407407407408, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.007407407407407408, 0.012962962962962963, 0.005555555555555556, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.007407407407407408, 0.007407407407407408, 0.005555555555555556, 0.012962962962962963, 0.005555555555555556, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.009259259259259259, 0.005555555555555556, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112],
            "cec7-2014": [0.007407407407407408, 0.007407407407407408, 0.007407407407407408, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.009259259259259259, 0.005555555555555556, 0.005555555555555556, 0.005555555555555556, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.007407407407407408, 0.007407407407407408, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.009259259259259259, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.009259259259259259, 0.009259259259259259, 0.005555555555555556, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.007407407407407408, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.012962962962962963, 0.005555555555555556, 0.005555555555555556, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259],
            "cec8-2014": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            "cec9-2014": [0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.005555555555555556, 0.011111111111111112, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.007407407407407408, 0.011111111111111112, 0.007407407407407408, 0.007407407407407408, 0.007407407407407408, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.005555555555555556, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.009259259259259259, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112],
            "cec10-2014": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            "cec11-2014": [0.005555555555555556, 0.005555555555555556, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.011111111111111112, 0.005555555555555556, 0.007407407407407408, 0.009259259259259259, 0.011111111111111112, 0.007407407407407408, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.007407407407407408, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.005555555555555556, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.007407407407407408, 0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408],
            "cec12-2014": [0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.007407407407407408, 0.009259259259259259, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.007407407407407408, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.005555555555555556, 0.007407407407407408, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.007407407407407408, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.005555555555555556],
            "cec13-2014": [0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.005555555555555556, 0.005555555555555556, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.005555555555555556, 0.007407407407407408, 0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.007407407407407408, 0.011111111111111112, 0.007407407407407408, 0.005555555555555556, 0.009259259259259259, 0.007407407407407408, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.005555555555555556, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.011111111111111112, 0.009259259259259259, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.007407407407407408, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.005555555555555556, 0.009259259259259259, 0.007407407407407408, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112],
            "cec14-2014": [0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.009259259259259259, 0.009259259259259259, 0.005555555555555556, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.005555555555555556, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.012962962962962963, 0.007407407407407408, 0.009259259259259259, 0.007407407407407408, 0.005555555555555556, 0.011111111111111112, 0.007407407407407408, 0.005555555555555556, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.007407407407407408, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.007407407407407408, 0.009259259259259259, 0.011111111111111112, 0.007407407407407408, 0.007407407407407408, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.007407407407407408, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408],
            "cec15-2014": [0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.005555555555555556, 0.009259259259259259, 0.009259259259259259, 0.009259259259259259, 0.005555555555555556, 0.009259259259259259, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.005555555555555556, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.009259259259259259, 0.005555555555555556, 0.009259259259259259, 0.012962962962962963, 0.005555555555555556, 0.009259259259259259, 0.007407407407407408, 0.005555555555555556, 0.011111111111111112, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.007407407407407408, 0.007407407407407408, 0.012962962962962963, 0.007407407407407408, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.007407407407407408, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963],
            "cec16-2014": [0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.005555555555555556, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.007407407407407408, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.005555555555555556, 0.011111111111111112, 0.007407407407407408, 0.007407407407407408, 0.011111111111111112, 0.009259259259259259, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.005555555555555556, 0.007407407407407408, 0.007407407407407408, 0.011111111111111112, 0.007407407407407408, 0.007407407407407408, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963],
            "cec17-2014": [0.009302325581395349, 0.0069767441860465115, 0.009302325581395349, 0.0069767441860465115, 0.0069767441860465115, 0.0069767441860465115, 0.009302325581395349, 0.013953488372093023, 0.009302325581395349, 0.013953488372093023, 0.013953488372093023, 0.009302325581395349, 0.009302325581395349, 0.013953488372093023, 0.0069767441860465115, 0.009302325581395349, 0.009302325581395349, 0.0069767441860465115, 0.009302325581395349, 0.0069767441860465115, 0.013953488372093023, 0.009302325581395349, 0.0069767441860465115, 0.013953488372093023, 0.0069767441860465115, 0.0069767441860465115, 0.009302325581395349, 0.0069767441860465115, 0.009302325581395349, 0.009302325581395349, 0.0069767441860465115, 0.013953488372093023, 0.009302325581395349, 0.009302325581395349, 0.009302325581395349, 0.0069767441860465115, 0.009302325581395349, 0.0069767441860465115, 0.009302325581395349, 0.0069767441860465115, 0.013953488372093023, 0.009302325581395349, 0.009302325581395349, 0.0069767441860465115, 0.013953488372093023, 0.009302325581395349, 0.0069767441860465115, 0.013953488372093023, 0.0069767441860465115, 0.009302325581395349, 0.0069767441860465115, 0.0069767441860465115, 0.009302325581395349, 0.009302325581395349, 0.013953488372093023, 0.013953488372093023, 0.009302325581395349, 0.013953488372093023, 0.009302325581395349, 0.009302325581395349, 0.0069767441860465115, 0.0069767441860465115, 0.009302325581395349, 0.009302325581395349, 0.009302325581395349, 0.013953488372093023, 0.013953488372093023, 0.009302325581395349, 0.013953488372093023, 0.0069767441860465115, 0.013953488372093023, 0.013953488372093023, 0.013953488372093023, 0.009302325581395349, 0.009302325581395349, 0.013953488372093023, 0.009302325581395349, 0.009302325581395349, 0.013953488372093023, 0.013953488372093023, 0.013953488372093023, 0.009302325581395349, 0.009302325581395349, 0.0069767441860465115, 0.0069767441860465115, 0.0069767441860465115, 0.013953488372093023, 0.013953488372093023, 0.013953488372093023, 0.013953488372093023, 0.0069767441860465115, 0.009302325581395349, 0.0069767441860465115, 0.013953488372093023, 0.013953488372093023, 0.009302325581395349, 0.013953488372093023, 0.0069767441860465115, 0.0069767441860465115, 0.009302325581395349],
            "cec18-2014": [0.015584415584415584, 0.01038961038961039, 0.01038961038961039, 0.01038961038961039, 0.015584415584415584, 0.015584415584415584, 0.015584415584415584, 0.015584415584415584, 0.003896103896103896, 0.003896103896103896, 0.01038961038961039, 0.015584415584415584, 0.015584415584415584, 0.015584415584415584, 0.003896103896103896, 0.01038961038961039, 0.003896103896103896, 0.01038961038961039, 0.003896103896103896, 0.003896103896103896, 0.01038961038961039, 0.01038961038961039, 0.015584415584415584, 0.01038961038961039, 0.015584415584415584, 0.01038961038961039, 0.01038961038961039, 0.015584415584415584, 0.015584415584415584, 0.003896103896103896, 0.01038961038961039, 0.01038961038961039, 0.01038961038961039, 0.01038961038961039, 0.015584415584415584, 0.01038961038961039, 0.003896103896103896, 0.01038961038961039, 0.01038961038961039, 0.01038961038961039, 0.003896103896103896, 0.003896103896103896, 0.01038961038961039, 0.01038961038961039, 0.01038961038961039, 0.01038961038961039, 0.01038961038961039, 0.01038961038961039, 0.003896103896103896, 0.003896103896103896, 0.015584415584415584, 0.003896103896103896, 0.003896103896103896, 0.01038961038961039, 0.003896103896103896, 0.015584415584415584, 0.003896103896103896, 0.01038961038961039, 0.015584415584415584, 0.01038961038961039, 0.015584415584415584, 0.003896103896103896, 0.015584415584415584, 0.015584415584415584, 0.003896103896103896, 0.003896103896103896, 0.003896103896103896, 0.015584415584415584, 0.003896103896103896, 0.015584415584415584, 0.01038961038961039, 0.015584415584415584, 0.003896103896103896, 0.003896103896103896, 0.01038961038961039, 0.015584415584415584, 0.003896103896103896, 0.01038961038961039, 0.003896103896103896, 0.015584415584415584, 0.003896103896103896, 0.015584415584415584, 0.015584415584415584, 0.01038961038961039, 0.003896103896103896, 0.01038961038961039, 0.015584415584415584, 0.01038961038961039, 0.01038961038961039, 0.01038961038961039, 0.015584415584415584, 0.015584415584415584, 0.003896103896103896, 0.003896103896103896, 0.015584415584415584, 0.01038961038961039, 0.01038961038961039, 0.01038961038961039, 0.01038961038961039, 0.003896103896103896],
            "cec19-2014": [0.015508021390374332, 0.0071301247771836, 0.0035650623885918, 0.015508021390374332, 0.0071301247771836, 0.0071301247771836, 0.015508021390374332, 0.0071301247771836, 0.0071301247771836, 0.0106951871657754, 0.015508021390374332, 0.0106951871657754, 0.0106951871657754, 0.015508021390374332, 0.0106951871657754, 0.015508021390374332, 0.0071301247771836, 0.015508021390374332, 0.015508021390374332, 0.0071301247771836, 0.0106951871657754, 0.015508021390374332, 0.0071301247771836, 0.0106951871657754, 0.0106951871657754, 0.0071301247771836, 0.0071301247771836, 0.0035650623885918, 0.0071301247771836, 0.0035650623885918, 0.0035650623885918, 0.0035650623885918, 0.0106951871657754, 0.0035650623885918, 0.0071301247771836, 0.0106951871657754, 0.0035650623885918, 0.0106951871657754, 0.015508021390374332, 0.0106951871657754, 0.0071301247771836, 0.015508021390374332, 0.015508021390374332, 0.0035650623885918, 0.0106951871657754, 0.015508021390374332, 0.0071301247771836, 0.0106951871657754, 0.015508021390374332, 0.015508021390374332, 0.0035650623885918, 0.015508021390374332, 0.0106951871657754, 0.0071301247771836, 0.015508021390374332, 0.0071301247771836, 0.0035650623885918, 0.0071301247771836, 0.0106951871657754, 0.0106951871657754, 0.0106951871657754, 0.0106951871657754, 0.015508021390374332, 0.015508021390374332, 0.0035650623885918, 0.0106951871657754, 0.0106951871657754, 0.015508021390374332, 0.0035650623885918, 0.0106951871657754, 0.015508021390374332, 0.0106951871657754, 0.015508021390374332, 0.015508021390374332, 0.0106951871657754, 0.0035650623885918, 0.0071301247771836, 0.0106951871657754, 0.0035650623885918, 0.0035650623885918, 0.015508021390374332, 0.015508021390374332, 0.0035650623885918, 0.015508021390374332, 0.0035650623885918, 0.0035650623885918, 0.0106951871657754, 0.0035650623885918, 0.015508021390374332, 0.0106951871657754, 0.0071301247771836, 0.015508021390374332, 0.015508021390374332, 0.0035650623885918, 0.0071301247771836, 0.0106951871657754, 0.0106951871657754, 0.015508021390374332, 0.0106951871657754, 0.0106951871657754],
            "cec20-2014": [0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.014285714285714285, 0.0035714285714285713, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.014285714285714285, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.0035714285714285713, 0.014285714285714285, 0.010714285714285714, 0.010714285714285714, 0.0035714285714285713, 0.010714285714285714, 0.014285714285714285, 0.014285714285714285, 0.0035714285714285713, 0.0035714285714285713, 0.010714285714285714, 0.010714285714285714, 0.0035714285714285713, 0.0035714285714285713, 0.010714285714285714, 0.010714285714285714, 0.014285714285714285, 0.010714285714285714, 0.0035714285714285713, 0.010714285714285714, 0.0035714285714285713, 0.010714285714285714, 0.0035714285714285713, 0.010714285714285714, 0.0035714285714285713, 0.0035714285714285713, 0.010714285714285714, 0.014285714285714285, 0.010714285714285714, 0.010714285714285714, 0.014285714285714285, 0.0035714285714285713, 0.0035714285714285713, 0.010714285714285714, 0.010714285714285714, 0.014285714285714285, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.0035714285714285713, 0.0035714285714285713, 0.010714285714285714, 0.0035714285714285713, 0.014285714285714285, 0.010714285714285714, 0.010714285714285714, 0.014285714285714285, 0.014285714285714285, 0.014285714285714285, 0.014285714285714285, 0.010714285714285714, 0.010714285714285714, 0.014285714285714285, 0.010714285714285714, 0.010714285714285714, 0.014285714285714285, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.014285714285714285, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.0035714285714285713, 0.010714285714285714, 0.010714285714285714, 0.014285714285714285, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.014285714285714285, 0.014285714285714285, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.0035714285714285713, 0.010714285714285714, 0.010714285714285714, 0.0035714285714285713],
            "cec21-2014": [0.007075471698113208, 0.0047169811320754715, 0.013443396226415095, 0.0047169811320754715, 0.013443396226415095, 0.018867924528301886, 0.0047169811320754715, 0.007075471698113208, 0.0047169811320754715, 0.013443396226415095, 0.0047169811320754715, 0.007075471698113208, 0.007075471698113208, 0.018867924528301886, 0.0047169811320754715, 0.013443396226415095, 0.0047169811320754715, 0.0047169811320754715, 0.013443396226415095, 0.013443396226415095, 0.013443396226415095, 0.013443396226415095, 0.007075471698113208, 0.0047169811320754715, 0.007075471698113208, 0.013443396226415095, 0.0047169811320754715, 0.007075471698113208, 0.018867924528301886, 0.018867924528301886, 0.007075471698113208, 0.018867924528301886, 0.013443396226415095, 0.0047169811320754715, 0.018867924528301886, 0.007075471698113208, 0.0047169811320754715, 0.007075471698113208, 0.013443396226415095, 0.0047169811320754715, 0.0047169811320754715, 0.0047169811320754715, 0.0047169811320754715, 0.007075471698113208, 0.018867924528301886, 0.007075471698113208, 0.018867924528301886, 0.018867924528301886, 0.013443396226415095, 0.013443396226415095, 0.007075471698113208, 0.018867924528301886, 0.018867924528301886, 0.013443396226415095, 0.018867924528301886, 0.0047169811320754715, 0.018867924528301886, 0.013443396226415095, 0.018867924528301886, 0.013443396226415095, 0.018867924528301886, 0.0047169811320754715, 0.0047169811320754715, 0.0047169811320754715, 0.007075471698113208, 0.0047169811320754715, 0.007075471698113208, 0.0047169811320754715, 0.007075471698113208, 0.0047169811320754715, 0.007075471698113208, 0.007075471698113208, 0.007075471698113208, 0.007075471698113208, 0.007075471698113208, 0.007075471698113208, 0.0047169811320754715, 0.018867924528301886, 0.0047169811320754715, 0.0047169811320754715, 0.018867924528301886, 0.0047169811320754715, 0.013443396226415095, 0.007075471698113208, 0.007075471698113208, 0.007075471698113208, 0.018867924528301886, 0.013443396226415095, 0.007075471698113208, 0.018867924528301886, 0.018867924528301886, 0.007075471698113208, 0.0047169811320754715, 0.007075471698113208, 0.0047169811320754715, 0.013443396226415095, 0.0047169811320754715, 0.013443396226415095, 0.007075471698113208, 0.007075471698113208],
            "cec22-2014": [0.013636363636363636, 0.013636363636363636, 0.00909090909090909, 0.013636363636363636, 0.004545454545454545, 0.00909090909090909, 0.013636363636363636, 0.00909090909090909, 0.00909090909090909, 0.00909090909090909, 0.013636363636363636, 0.013636363636363636, 0.00909090909090909, 0.013636363636363636, 0.013636363636363636, 0.004545454545454545, 0.00909090909090909, 0.013636363636363636, 0.00909090909090909, 0.004545454545454545, 0.013636363636363636, 0.00909090909090909, 0.00909090909090909, 0.013636363636363636, 0.013636363636363636, 0.00909090909090909, 0.013636363636363636, 0.004545454545454545, 0.013636363636363636, 0.013636363636363636, 0.004545454545454545, 0.004545454545454545, 0.004545454545454545, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.004545454545454545, 0.004545454545454545, 0.004545454545454545, 0.004545454545454545, 0.00909090909090909, 0.004545454545454545, 0.00909090909090909, 0.004545454545454545, 0.00909090909090909, 0.013636363636363636, 0.004545454545454545, 0.004545454545454545, 0.00909090909090909, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.00909090909090909, 0.004545454545454545, 0.013636363636363636, 0.013636363636363636, 0.004545454545454545, 0.004545454545454545, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.00909090909090909, 0.004545454545454545, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.004545454545454545, 0.013636363636363636, 0.004545454545454545, 0.004545454545454545, 0.013636363636363636, 0.004545454545454545, 0.013636363636363636, 0.00909090909090909, 0.004545454545454545, 0.004545454545454545, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.004545454545454545, 0.004545454545454545, 0.004545454545454545, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.004545454545454545, 0.004545454545454545, 0.013636363636363636, 0.00909090909090909, 0.00909090909090909, 0.013636363636363636],
            "cec23-2014": [0.008554842652001222, 0.009318667888787045, 0.010235258172930034, 0.01008249312557287, 0.009318667888787045, 0.009624197983501375, 0.012679498930644668, 0.011762908646501681, 0.01237396883593034, 0.010540788267644362, 0.010235258172930034, 0.010235258172930034, 0.00717995722578674, 0.00977696303085854, 0.009929728078215704, 0.009013137794072716, 0.011762908646501681, 0.010235258172930034, 0.011151848457073022, 0.008707607699358386, 0.008707607699358386, 0.008707607699358386, 0.011457378551787351, 0.011457378551787351, 0.007485487320501069, 0.010235258172930034, 0.011762908646501681, 0.010540788267644362, 0.010540788267644362, 0.009013137794072716, 0.009471432936144211, 0.009318667888787045, 0.011762908646501681, 0.011457378551787351, 0.010235258172930034, 0.012679498930644668, 0.010235258172930034, 0.011762908646501681, 0.009013137794072716, 0.010235258172930034, 0.01008249312557287, 0.009929728078215704, 0.010846318362358692, 0.009318667888787045, 0.009318667888787045, 0.010540788267644362, 0.007791017415215398, 0.01206843874121601, 0.01237396883593034, 0.009318667888787045, 0.010540788267644362, 0.01206843874121601, 0.009318667888787045, 0.011762908646501681, 0.009624197983501375, 0.007791017415215398, 0.010846318362358692, 0.008249312557286892, 0.009013137794072716, 0.010999083409715857, 0.009013137794072716, 0.009013137794072716, 0.010846318362358692, 0.011457378551787351, 0.011151848457073022, 0.010235258172930034, 0.010235258172930034, 0.011762908646501681, 0.011762908646501681, 0.011151848457073022, 0.007791017415215398, 0.011762908646501681, 0.009318667888787045, 0.0068744271310724105, 0.010846318362358692, 0.007791017415215398, 0.007791017415215398, 0.010540788267644362, 0.010540788267644362, 0.010235258172930034, 0.008402077604644058, 0.009318667888787045, 0.009318667888787045, 0.007791017415215398, 0.009929728078215704, 0.01206843874121601, 0.007485487320501069, 0.008249312557286892, 0.010693553315001528, 0.007791017415215398, 0.008096547509929728, 0.011151848457073022, 0.009471432936144211, 0.008402077604644058, 0.009318667888787045, 0.009318667888787045, 0.011762908646501681, 0.011457378551787351, 0.008707607699358386, 0.010540788267644362],
            "cec24-2014": [0.011094224924012158, 0.008662613981762919, 0.006838905775075988, 0.005623100303951368, 0.008054711246200608, 0.011702127659574468, 0.011094224924012158, 0.009270516717325228, 0.012917933130699088, 0.008054711246200608, 0.008054711246200608, 0.012917933130699088, 0.011702127659574468, 0.011094224924012158, 0.009270516717325228, 0.009270516717325228, 0.008054711246200608, 0.011702127659574468, 0.008054711246200608, 0.012310030395136779, 0.007446808510638298, 0.010486322188449849, 0.008054711246200608, 0.009878419452887538, 0.007446808510638298, 0.006838905775075988, 0.011702127659574468, 0.011094224924012158, 0.007446808510638298, 0.011702127659574468, 0.011094224924012158, 0.011094224924012158, 0.009878419452887538, 0.012310030395136779, 0.011702127659574468, 0.010486322188449849, 0.010486322188449849, 0.006838905775075988, 0.010486322188449849, 0.009270516717325228, 0.010486322188449849, 0.008054711246200608, 0.008662613981762919, 0.008054711246200608, 0.008054711246200608, 0.012917933130699088, 0.006231003039513678, 0.009878419452887538, 0.012310030395136779, 0.009878419452887538, 0.011094224924012158, 0.011702127659574468, 0.011094224924012158, 0.007446808510638298, 0.011702127659574468, 0.011094224924012158, 0.010486322188449849, 0.011094224924012158, 0.006231003039513678, 0.008054711246200608, 0.007446808510638298, 0.012917933130699088, 0.009878419452887538, 0.011702127659574468, 0.009878419452887538, 0.010486322188449849, 0.012917933130699088, 0.009270516717325228, 0.009270516717325228, 0.010486322188449849, 0.010486322188449849, 0.006231003039513678, 0.011094224924012158, 0.012310030395136779, 0.010486322188449849, 0.008662613981762919, 0.012310030395136779, 0.008662613981762919, 0.006838905775075988, 0.011702127659574468, 0.012310030395136779, 0.010486322188449849, 0.010486322188449849, 0.009878419452887538, 0.009270516717325228, 0.012917933130699088, 0.010486322188449849, 0.012917933130699088, 0.009270516717325228, 0.008662613981762919, 0.012917933130699088, 0.011702127659574468, 0.011702127659574468, 0.009270516717325228, 0.007446808510638298, 0.008054711246200608, 0.010486322188449849, 0.008662613981762919, 0.012917933130699088, 0.011094224924012158],
            "cec25-2014": [0.008333333333333333, 0.012962962962962963, 0.010185185185185186, 0.010185185185185186, 0.011574074074074073, 0.009259259259259259, 0.007407407407407408, 0.00787037037037037, 0.008796296296296297, 0.011111111111111112, 0.009722222222222222, 0.011574074074074073, 0.0125, 0.009722222222222222, 0.010185185185185186, 0.010648148148148148, 0.008796296296296297, 0.010648148148148148, 0.009259259259259259, 0.00787037037037037, 0.009722222222222222, 0.009722222222222222, 0.011574074074074073, 0.012037037037037037, 0.008796296296296297, 0.009259259259259259, 0.011111111111111112, 0.010648148148148148, 0.010648148148148148, 0.0125, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.00787037037037037, 0.008796296296296297, 0.006481481481481481, 0.009722222222222222, 0.008796296296296297, 0.008796296296296297, 0.010648148148148148, 0.010648148148148148, 0.008796296296296297, 0.008796296296296297, 0.011574074074074073, 0.00787037037037037, 0.011111111111111112, 0.010185185185185186, 0.011111111111111112, 0.006944444444444444, 0.011111111111111112, 0.011111111111111112, 0.009259259259259259, 0.0125, 0.011111111111111112, 0.009722222222222222, 0.008333333333333333, 0.009259259259259259, 0.011111111111111112, 0.008796296296296297, 0.008333333333333333, 0.008796296296296297, 0.009722222222222222, 0.0060185185185185185, 0.011111111111111112, 0.012962962962962963, 0.011574074074074073, 0.010648148148148148, 0.010648148148148148, 0.00787037037037037, 0.012037037037037037, 0.00787037037037037, 0.011574074074074073, 0.009722222222222222, 0.012037037037037037, 0.010185185185185186, 0.012037037037037037, 0.008333333333333333, 0.011111111111111112, 0.008333333333333333, 0.010185185185185186, 0.011574074074074073, 0.011111111111111112, 0.010648148148148148, 0.006481481481481481, 0.006481481481481481, 0.010648148148148148, 0.011111111111111112, 0.012037037037037037, 0.009722222222222222, 0.010185185185185186, 0.008796296296296297, 0.010185185185185186, 0.00787037037037037, 0.012037037037037037, 0.009259259259259259, 0.010648148148148148, 0.008333333333333333, 0.011574074074074073, 0.011111111111111112, 0.010648148148148148],
            "cec26-2014": [0.009722222222222222, 0.01087962962962963, 0.007407407407407408, 0.00949074074074074, 0.01087962962962963, 0.006481481481481481, 0.010648148148148148, 0.011574074074074073, 0.00949074074074074, 0.009259259259259259, 0.00949074074074074, 0.01087962962962963, 0.011574074074074073, 0.009722222222222222, 0.008101851851851851, 0.009722222222222222, 0.01087962962962963, 0.010185185185185186, 0.009722222222222222, 0.009259259259259259, 0.010648148148148148, 0.008333333333333333, 0.011574074074074073, 0.006944444444444444, 0.008333333333333333, 0.009027777777777777, 0.010416666666666666, 0.012037037037037037, 0.009722222222222222, 0.010648148148148148, 0.011111111111111112, 0.008564814814814815, 0.009953703703703704, 0.008796296296296297, 0.01087962962962963, 0.009259259259259259, 0.008101851851851851, 0.008796296296296297, 0.010416666666666666, 0.00787037037037037, 0.008796296296296297, 0.00949074074074074, 0.010185185185185186, 0.011574074074074073, 0.011111111111111112, 0.01087962962962963, 0.010185185185185186, 0.009027777777777777, 0.011111111111111112, 0.01087962962962963, 0.008796296296296297, 0.011111111111111112, 0.010416666666666666, 0.008101851851851851, 0.011111111111111112, 0.011805555555555555, 0.010648148148148148, 0.009953703703703704, 0.01087962962962963, 0.008796296296296297, 0.008333333333333333, 0.008333333333333333, 0.01087962962962963, 0.011111111111111112, 0.008796296296296297, 0.009259259259259259, 0.010648148148148148, 0.00949074074074074, 0.011574074074074073, 0.008333333333333333, 0.012037037037037037, 0.012037037037037037, 0.00949074074074074, 0.010648148148148148, 0.00787037037037037, 0.00949074074074074, 0.011111111111111112, 0.011805555555555555, 0.011111111111111112, 0.011805555555555555, 0.008333333333333333, 0.009027777777777777, 0.010648148148148148, 0.01087962962962963, 0.01087962962962963, 0.010416666666666666, 0.00949074074074074, 0.011574074074074073, 0.012037037037037037, 0.011805555555555555, 0.011574074074074073, 0.010185185185185186, 0.008564814814814815, 0.010416666666666666, 0.009722222222222222, 0.009953703703703704, 0.009722222222222222, 0.008564814814814815, 0.01087962962962963, 0.00949074074074074],
            "cec27-2014": [0.011522633744855968, 0.010905349794238683, 0.011522633744855968, 0.010082304526748971, 0.010699588477366255, 0.010082304526748971, 0.010493827160493827, 0.00823045267489712, 0.008436213991769548, 0.008024691358024692, 0.009465020576131687, 0.00905349794238683, 0.01213991769547325, 0.010082304526748971, 0.008436213991769548, 0.010699588477366255, 0.007818930041152264, 0.011111111111111112, 0.01131687242798354, 0.009670781893004115, 0.010082304526748971, 0.007407407407407408, 0.011728395061728396, 0.01131687242798354, 0.00905349794238683, 0.009670781893004115, 0.009876543209876543, 0.008436213991769548, 0.010905349794238683, 0.011522633744855968, 0.00905349794238683, 0.009465020576131687, 0.008024691358024692, 0.008847736625514403, 0.009465020576131687, 0.010699588477366255, 0.011728395061728396, 0.012757201646090535, 0.007818930041152264, 0.008436213991769548, 0.007407407407407408, 0.010493827160493827, 0.007613168724279836, 0.009465020576131687, 0.011522633744855968, 0.012551440329218106, 0.009670781893004115, 0.007613168724279836, 0.010082304526748971, 0.00905349794238683, 0.011522633744855968, 0.01131687242798354, 0.008024691358024692, 0.011934156378600824, 0.012757201646090535, 0.011111111111111112, 0.009465020576131687, 0.01131687242798354, 0.009465020576131687, 0.00823045267489712, 0.012757201646090535, 0.011111111111111112, 0.00823045267489712, 0.008641975308641974, 0.010082304526748971, 0.010493827160493827, 0.01131687242798354, 0.009465020576131687, 0.006584362139917695, 0.008641975308641974, 0.0102880658436214, 0.01213991769547325, 0.010493827160493827, 0.011728395061728396, 0.009465020576131687, 0.011111111111111112, 0.009259259259259259, 0.008436213991769548, 0.010082304526748971, 0.011111111111111112, 0.010493827160493827, 0.010905349794238683, 0.0102880658436214, 0.009259259259259259, 0.010905349794238683, 0.0102880658436214, 0.008024691358024692, 0.010905349794238683, 0.010905349794238683, 0.010905349794238683, 0.010493827160493827, 0.008436213991769548, 0.009465020576131687, 0.011522633744855968, 0.009465020576131687, 0.009465020576131687, 0.010493827160493827, 0.008847736625514403, 0.009876543209876543, 0.010905349794238683],
            "cec28-2014": [0.009876543209876543, 0.008436213991769548, 0.008436213991769548, 0.009465020576131687, 0.012757201646090535, 0.010699588477366255, 0.009670781893004115, 0.011934156378600824, 0.010082304526748971, 0.011728395061728396, 0.008847736625514403, 0.010082304526748971, 0.011522633744855968, 0.009876543209876543, 0.006790123456790123, 0.0102880658436214, 0.009670781893004115, 0.01131687242798354, 0.011111111111111112, 0.009259259259259259, 0.009670781893004115, 0.011111111111111112, 0.009876543209876543, 0.008847736625514403, 0.011728395061728396, 0.010493827160493827, 0.0102880658436214, 0.00720164609053498, 0.009876543209876543, 0.008847736625514403, 0.010905349794238683, 0.00905349794238683, 0.010905349794238683, 0.008641975308641974, 0.010082304526748971, 0.009670781893004115, 0.011111111111111112, 0.01131687242798354, 0.010905349794238683, 0.011111111111111112, 0.010699588477366255, 0.009259259259259259, 0.009259259259259259, 0.011728395061728396, 0.008024691358024692, 0.010082304526748971, 0.008847736625514403, 0.010082304526748971, 0.011728395061728396, 0.010082304526748971, 0.011522633744855968, 0.011728395061728396, 0.009465020576131687, 0.011728395061728396, 0.009876543209876543, 0.0102880658436214, 0.010493827160493827, 0.010905349794238683, 0.00905349794238683, 0.010082304526748971, 0.009876543209876543, 0.00823045267489712, 0.011728395061728396, 0.0102880658436214, 0.008436213991769548, 0.010699588477366255, 0.009465020576131687, 0.010699588477366255, 0.008847736625514403, 0.01131687242798354, 0.009876543209876543, 0.008436213991769548, 0.00823045267489712, 0.00905349794238683, 0.009465020576131687, 0.00823045267489712, 0.009670781893004115, 0.010699588477366255, 0.007818930041152264, 0.010905349794238683, 0.00905349794238683, 0.010082304526748971, 0.009876543209876543, 0.00905349794238683, 0.0102880658436214, 0.009259259259259259, 0.011934156378600824, 0.008641975308641974, 0.008641975308641974, 0.012551440329218106, 0.010493827160493827, 0.00905349794238683, 0.009465020576131687, 0.010082304526748971, 0.007407407407407408, 0.010493827160493827, 0.009465020576131687, 0.011934156378600824, 0.011522633744855968, 0.0102880658436214],
            "cec29-2014": [0.011150047785919083, 0.01083147499203568, 0.010035043007327174, 0.00796431984708506, 0.005734310289901242, 0.010194329404268876, 0.015291494106403312, 0.011150047785919083, 0.005734310289901242, 0.012105766167569289, 0.012742911755336095, 0.015610066900286716, 0.008601465434851864, 0.010194329404268876, 0.010194329404268876, 0.007008601465434852, 0.014335775724753107, 0.011150047785919083, 0.007486460656259956, 0.007645747053201656, 0.010194329404268876, 0.008601465434851864, 0.011150047785919083, 0.010353615801210577, 0.007008601465434852, 0.009238611022618668, 0.01083147499203568, 0.00828289264096846, 0.0133800573431029, 0.011468620579802484, 0.008920038228735267, 0.0136986301369863, 0.003822873526600828, 0.012742911755336095, 0.010194329404268876, 0.00796431984708506, 0.011150047785919083, 0.009557183816502071, 0.009238611022618668, 0.007008601465434852, 0.008601465434851864, 0.011627906976744186, 0.011150047785919083, 0.011150047785919083, 0.010194329404268876, 0.012424338961452692, 0.005734310289901242, 0.012105766167569289, 0.005097164702134438, 0.0136986301369863, 0.011468620579802484, 0.006212169480726346, 0.0133800573431029, 0.007645747053201656, 0.008601465434851864, 0.0135393437400446, 0.009238611022618668, 0.010194329404268876, 0.012742911755336095, 0.011787193373685887, 0.011787193373685887, 0.010194329404268876, 0.01083147499203568, 0.005734310289901242, 0.009238611022618668, 0.00669002867155145, 0.007645747053201656, 0.009557183816502071, 0.009557183816502071, 0.008920038228735267, 0.00669002867155145, 0.011787193373685887, 0.011150047785919083, 0.0063714558776680474, 0.014017202930869704, 0.011150047785919083, 0.007645747053201656, 0.013061484549219496, 0.007645747053201656, 0.014176489327811405, 0.00796431984708506, 0.014335775724753107, 0.012742911755336095, 0.007645747053201656, 0.006212169480726346, 0.00796431984708506, 0.014335775724753107, 0.008920038228735267, 0.011468620579802484, 0.014335775724753107, 0.0133800573431029, 0.009238611022618668, 0.012583625358394393, 0.0063714558776680474, 0.008601465434851864, 0.010194329404268876, 0.009716470213443773, 0.003822873526600828, 0.008601465434851864, 0.011150047785919083],
            "cec30-2014": [0.008541600759253401, 0.012337867763366024, 0.011072445428661816, 0.011072445428661816, 0.008225245175577349, 0.013603290098070231, 0.008857956342929452, 0.012179689971527997, 0.008541600759253401, 0.010439734261309713, 0.007908889591901298, 0.007276178424549193, 0.013919645681746282, 0.009174311926605505, 0.012337867763366024, 0.010597912053147737, 0.010439734261309713, 0.009174311926605505, 0.01249604555520405, 0.01012337867763366, 0.009490667510281556, 0.009807023093957609, 0.013919645681746282, 0.005378044922492882, 0.013286934514394179, 0.010439734261309713, 0.010439734261309713, 0.009174311926605505, 0.008225245175577349, 0.010756089844985764, 0.009807023093957609, 0.009490667510281556, 0.011388801012337867, 0.009807023093957609, 0.011072445428661816, 0.010756089844985764, 0.01091426763682379, 0.013128756722556154, 0.006643467257197089, 0.008225245175577349, 0.007276178424549193, 0.012337867763366024, 0.007276178424549193, 0.01170515659601392, 0.008225245175577349, 0.007592534008225245, 0.009174311926605505, 0.008225245175577349, 0.014236001265422335, 0.0060107560898449855, 0.012021512179689971, 0.01012337867763366, 0.012654223347042075, 0.009174311926605505, 0.008857956342929452, 0.009965200885795633, 0.011546978804175894, 0.012021512179689971, 0.009174311926605505, 0.008541600759253401, 0.010439734261309713, 0.013286934514394179, 0.01012337867763366, 0.008225245175577349, 0.008541600759253401, 0.007908889591901298, 0.008541600759253401, 0.008225245175577349, 0.008541600759253401, 0.005378044922492882, 0.01249604555520405, 0.011072445428661816, 0.013286934514394179, 0.006959822840873141, 0.008541600759253401, 0.01170515659601392, 0.010439734261309713, 0.009807023093957609, 0.01249604555520405, 0.008225245175577349, 0.007908889591901298, 0.008857956342929452, 0.013286934514394179, 0.009490667510281556, 0.014552356849098386, 0.010756089844985764, 0.011072445428661816, 0.009807023093957609, 0.008225245175577349, 0.009965200885795633, 0.007908889591901298, 0.008225245175577349, 0.009490667510281556, 0.010756089844985764, 0.010439734261309713, 0.010756089844985764, 0.011072445428661816, 0.012970578930718128, 0.005378044922492882, 0.008225245175577349],            
        },

    }

    cur_degree = np.array(var_degree_obj_func[dim][func_name])
    return cur_degree # (dim, )
    

def calVarDegree():

    dim = 30
    fdir = './input_data'
    dataset = '2014' # 'classical set should set ''
 
    cf_num = {
        # For all dimensions in CEC-2014
        'cec23-2014': [1, 2, 3, 4],
        'cec24-2014': [2, 3],
        'cec25-2014': [1, 2, 3],
        'cec26-2014': [1, 2, 3, 4, 5],
        'cec27-2014': [1, 2, 3, 4, 5],
        'cec28-2014': [1, 2, 3, 4, 5],
        'cec29-2014': [1, 2, 3],
        'cec30-2014': [1, 2, 3],

    }
    if dataset == '2014':
        func_set = np.arange(1, 31)
    elif dataset == 'classical':
        raise NotImplementedError("The classical set should be implemented..")
    elif dataset == '2021':
        raise NotImplementedError("The 2021 set should be implemented..")
    else:
        raise ValueError("The dataset should be 2014, classical or 2021..")
    m_cnt = {}
    for func_num in func_set:

        func_name = 'cec{}-{}'.format(func_num, dataset)
        fpath = os.path.join(fdir, 'M_{}_D{}.txt'.format(func_num, dim))
        # read text file by row
        with open(fpath, 'r') as f:
            lines = f.readlines()
            m_mtx = []
            for line in lines:
                # split by space
                line = line.strip()
                line = re.sub(" +", " ", line)
                line = line.split(' ')
                line = np.array(line)
                line = line.astype(np.float32)
                m_mtx.append(line)
                
        m_mtx = np.array(m_mtx)
        row_len = m_mtx.shape[0]
        col_len = m_mtx.shape[1]
        if col_len != dim:
            raise ValueError("The column length is not equal to dim [{}-{}: {}]..".format(func_name, dim, col_len))

        if (func_num >= 17) and (func_num <= 22) and (dataset == '2014'):
            shuffle_fpath = os.path.join(fdir, 'shuffle_data_{}_D{}.txt'.format(func_num, dim))
            with open(shuffle_fpath, 'r') as f:
                lines = f.readlines()
                line = lines[0]
                line = line.strip()
                line = re.sub(" +", " ", line)
                line = re.sub("\t", " ", line)
                line = line.split(' ')
                line = np.array(line)
                shuffle_ay = line.astype(np.int) - 1
 
        elif (func_num >= 29) and (dataset == '2014'):
            shuffle_fpath = os.path.join(fdir, 'shuffle_data_{}_D{}.txt'.format(func_num, dim))
            with open(shuffle_fpath, 'r') as f:
                lines = f.readlines()
                line = lines[0]
                line = line.strip()
                line = re.sub(" +", " ", line)
                line = re.sub("\t", " ", line)
                line = line.split(' ')
                line = np.array(line)
                shuffle_ay = line.astype(np.int) - 1
        else:         
            pass


        if (func_num >= 23) and (dataset == '2014'):
            cf_idx_lst = np.array(cf_num[func_name]) - 1
            if func_name == "cec23-2014":
                total_comp = 5
                # [1, 2, 3, 4] comp has rotated M
                # comp1 
                cur_cf_range = np.arange(0*dim, 1*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0                
                c1 = [m1_flag[0] * 2]
                c2 = m1_flag[1:-1] * 3
                c2 = np.reshape(c2, (len(m1_flag)-2, dim))
                c3 = [m1_flag[-1] * 1]
                c4 = np.concatenate((c1, c2, c3), axis=0)
                comp1_cnt = np.sum(c4, axis=0)
                # comp2
                cur_cf_range = np.arange(1*dim, 2*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0     
                comp2_cnt = np.sum(m1_flag, axis=0)

                # comp3
                cur_cf_range = np.arange(2*dim, 3*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0 
                comp3_cnt = np.sum(m1_flag, axis=0)
                
                # comp4
                cur_cf_range = np.arange(3*dim, 4*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0
                comp4_cnt = np.sum(m1_flag, axis=0)                 
                
                # comp5
                comp5_cnt = np.ones(dim)
                m1_cnt = comp1_cnt + comp2_cnt + comp3_cnt + comp4_cnt + comp5_cnt

            elif func_name == "cec24-2014":
                total_comp = 3
                # [2, 3] comp has rotated M
                # comp1
                comp1_cnt = np.ones(dim)
                # comp2
                cur_cf_range = np.arange(1*dim, 2*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0
                comp2_cnt = np.sum(m1_flag*2, axis=0)
                # comp3
                cur_cf_range = np.arange(2*dim, 3*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0
                comp3_cnt = np.sum(m1_flag*4, axis=0)
                m1_cnt = comp1_cnt + comp2_cnt + comp3_cnt


            elif func_name == "cec25-2014":
                total_comp = 3
                # [1, 2, 3] comp has rotated M
                # comp1
                cur_cf_range = np.arange(0*dim, 1*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0
                comp1_cnt = np.sum(m1_flag, axis=0)
                # comp2
                cur_cf_range = np.arange(1*dim, 2*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0
                comp2_cnt = np.sum(m1_flag*2, axis=0)
                # comp3
                cur_cf_range = np.arange(2*dim, 3*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0
                comp3_cnt = np.sum(m1_flag, axis=0)
                m1_cnt = comp1_cnt + comp2_cnt + comp3_cnt

            elif func_name == "cec26-2014":
                total_comp = 5
                # [1, 2, 3, 4, 5] comp has rotated M
                # comp1
                cur_cf_range = np.arange(0*dim, 1*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0
                comp1_cnt = np.sum(m1_flag, axis=0)

                # comp2
                cur_cf_range = np.arange(1*dim, 2*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0
                comp2_cnt = np.sum(m1_flag*3, axis=0)
                
                # comp3
                cur_cf_range = np.arange(2*dim, 3*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0
                comp3_cnt = np.sum(m1_flag, axis=0)

                # comp4
                cur_cf_range = np.arange(3*dim, 4*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0
                comp4_cnt = np.sum(m1_flag, axis=0)

                # comp5
                cur_cf_range = np.arange(4*dim, 5*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0
                comp5_cnt = np.sum(m1_flag*2, axis=0)

                m1_cnt = comp1_cnt + comp2_cnt + comp3_cnt + comp4_cnt + comp5_cnt

            elif func_name == "cec27-2014":
                total_comp = 5
                # [1, 2, 3, 4, 5] comp has rotated M
                # comp1
                cur_cf_range = np.arange(0*dim, 1*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0
                comp1_cnt = np.sum(m1_flag*4, axis=0)

                # comp2
                cur_cf_range = np.arange(1*dim, 2*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0
                comp2_cnt = np.sum(m1_flag*2, axis=0)
                
                # comp3
                cur_cf_range = np.arange(2*dim, 3*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0
                comp3_cnt = np.sum(m1_flag, axis=0)

                # comp4
                cur_cf_range = np.arange(3*dim, 4*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0
                comp4_cnt = np.sum(m1_flag, axis=0)

                # comp5
                cur_cf_range = np.arange(4*dim, 5*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0
                comp5_cnt = np.sum(m1_flag, axis=0)

                m1_cnt = comp1_cnt + comp2_cnt + comp3_cnt + comp4_cnt + comp5_cnt

            elif func_name == "cec28-2014":
                total_comp = 5
                # [1, 2, 3, 4, 5] comp has rotated M
                # comp1
                cur_cf_range = np.arange(0*dim, 1*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0
                comp1_cnt = np.sum(m1_flag*2, axis=0)

                # comp2
                cur_cf_range = np.arange(1*dim, 2*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0
                comp2_cnt = np.sum(m1_flag*3, axis=0)
                
                # comp3
                cur_cf_range = np.arange(2*dim, 3*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0
                comp3_cnt = np.sum(m1_flag, axis=0)

                # comp4
                cur_cf_range = np.arange(3*dim, 4*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0
                comp4_cnt = np.sum(m1_flag*2, axis=0)

                # comp5
                cur_cf_range = np.arange(4*dim, 5*dim)
                m1 = m_mtx[cur_cf_range]
                m1_flag = m1 != 0
                comp5_cnt = np.sum(m1_flag, axis=0)

                m1_cnt = comp1_cnt + comp2_cnt + comp3_cnt + comp4_cnt + comp5_cnt

            elif func_name == "cec29-2014":
                total_comp = 3
                # [1, 2, 3] comp has rotated M
                # 有shuffle
                # comp1
                cur_cf_range = np.arange(0*dim, 1*dim)
                m1 = m_mtx[cur_cf_range]   
                m1_flag = m1 != 0             
                m1_flag = m1_flag[shuffle_ay[0*dim: 1*dim]]
                pt1 = int(np.ceil(dim*0.3))
                pt2 = pt1 + int(np.ceil(dim*0.3))
                c1 = np.sum(m1_flag[:pt1], axis=0)
                c2 = np.sum(m1_flag[pt1:pt2]*2, axis=0)
                c3 = np.sum(m1_flag[pt2:], axis=0)
                comp1_cnt = c1 + c2 + c3
                # comp2
                cur_cf_range = np.arange(1*dim, 2*dim)
                m1 = m_mtx[cur_cf_range]   
                m1_flag = m1 != 0     
                m1_flag = m1_flag[shuffle_ay[1*dim: 2*dim]]
                pt1 = int(np.ceil(dim*0.3))
                pt2 = pt1 + int(np.ceil(dim*0.3))
                c1 = np.sum(m1_flag[:pt1], axis=0)
                c2 = np.sum(m1_flag[pt1:pt2]*4, axis=0)
                c3 = np.sum(m1_flag[pt2:]*2, axis=0)
                comp2_cnt = c1 + c2 + c3

                # comp3
                cur_cf_range = np.arange(2*dim, 3*dim)
                m1 = m_mtx[cur_cf_range]   
                m1_flag = m1 != 0     
                m1_flag = m1_flag[shuffle_ay[2*dim: 3*dim]]
                pt1 = int(np.ceil(dim*0.2))
                pt2 = pt1 + int(np.ceil(dim*0.2))
                pt3 = pt2 + int(np.ceil(dim*0.3))

                c1 = np.sum(m1_flag[:pt1]*2, axis=0)
                c2 = np.sum(m1_flag[pt1:pt2], axis=0)
                
                m2_flag = m1_flag[pt2:pt3]
                c31 = [m2_flag[0] * 2]
                c32 = m2_flag[1:-1] * 3
                c32 = np.reshape(c32, (len(m2_flag)-2, dim))
                c33 = [m2_flag[-1] * 1]
                c34 = np.concatenate((c31, c32, c33), axis=0)
                c3 = np.sum(c34, axis=0)
                c4 = np.sum(m1_flag[pt3:]*2, axis=0)
                comp3_cnt = c1 + c2 + c3 + c4

                m1_cnt = comp1_cnt + comp2_cnt + comp3_cnt

            elif func_name == "cec30-2014":
                total_comp = 3
                # [1, 2, 3] comp has rotated M
                # 有shuffle
                # comp1
                cur_cf_range = np.arange(0*dim, 1*dim)
                m1 = m_mtx[cur_cf_range]   
                m1_flag = m1 != 0        
                m1_flag = m1_flag[shuffle_ay[0*dim: 1*dim]]
                pt1 = int(np.ceil(dim*0.2))
                pt2 = pt1 + int(np.ceil(dim*0.2))
                pt3 = pt2 + int(np.ceil(dim*0.3))

                c1 = np.sum(m1_flag[:pt1]*4, axis=0)
                c2 = np.sum(m1_flag[pt1:pt2], axis=0)
                c3 = np.sum(m1_flag[pt2:pt3]*2, axis=0)
                c4 = np.sum(m1_flag[pt3:]*2, axis=0)
                comp1_cnt = c1 + c2 + c3 + c4

                # comp2
                cur_cf_range = np.arange(1*dim, 2*dim)
                m1 = m_mtx[cur_cf_range]   
                m1_flag = m1 != 0     
                m1_flag = m1_flag[shuffle_ay[1*dim: 2*dim]]
                pt1 = int(np.ceil(dim*0.1))
                pt2 = pt1 + int(np.ceil(dim*0.2))
                pt3 = pt2 + int(np.ceil(dim*0.2))
                pt4 = pt3 + int(np.ceil(dim*0.2))

                c1 = np.sum(m1_flag[:pt1]*2, axis=0)
                c2 = np.sum(m1_flag[pt1:pt2]*4, axis=0)
                
                m2_flag = m1_flag[pt2:pt3]
                c31 = [m2_flag[0] * 2]
                c32 = m2_flag[1:-1] * 3
                c32 = np.reshape(c32, (len(m2_flag)-2, dim))
                c33 = [m2_flag[-1] * 1]
                c34 = np.concatenate((c31, c32, c33), axis=0)
                
                c3 = np.sum(c34, axis=0)
                c4 = np.sum(m1_flag[pt3:pt4], axis=0)
                c5 = np.sum(m1_flag[pt4:], axis=0)
                comp2_cnt = c1 + c2 + c3 + c4 + c5

                # comp3
                cur_cf_range = np.arange(2*dim, 3*dim)
                m1 = m_mtx[cur_cf_range]   
                m1_flag = m1 != 0 
                m1_flag = m1_flag[shuffle_ay[2*dim: 3*dim]]
                pt1 = int(np.ceil(dim*0.1))
                pt2 = pt1 + int(np.ceil(dim*0.2))
                pt3 = pt2 + int(np.ceil(dim*0.2))
                pt4 = pt3 + int(np.ceil(dim*0.2))
                c1 = np.sum(m1_flag[:pt1]*2, axis=0)
                c2 = np.sum(m1_flag[pt1:pt2]*3, axis=0)
                c3 = np.sum(m1_flag[pt2:pt3]*2, axis=0)
                c4 = np.sum(m1_flag[pt3:pt4], axis=0)
                c5 = np.sum(m1_flag[pt4:]*2, axis=0)
                comp3_cnt = c1 + c2 + c3 + c4 + c5

                m1_cnt = comp1_cnt + comp2_cnt + comp3_cnt
            else:
                raise ValueError("The function name is not in the list..")

        else:
            if row_len != dim:
                raise ValueError("The row length is not equal to dim [{}-{}: {}]..".format(func_name, dim, row_len))
            m1_flag = m_mtx != 0
            if func_name in ['cec1-2014', 'cec2-2014', 'cec3-2014', 'cec6-2014', 'cec11-2014']:
                m1_cnt = np.sum(m1_flag, axis=0)

            elif func_name == 'cec4-2014':
                c1 = [m1_flag[0] * 2]
                c2 = m1_flag[1:-1] * 3
                c2 = np.reshape(c2, (dim-2, dim))
                c3 = [m1_flag[-1] * 1]
                c4 = np.concatenate((c1, c2, c3), axis=0)
                m1_cnt = np.sum(c4, axis=0)

            elif func_name in ['cec5-2014', 'cec7-2014', 'cec9-2014', 'cec12-2014', 'cec15-2014', 'cec16-2014']:
                m1_cnt = np.sum(m1_flag*2, axis=0)

            elif func_name in ['cec8-2014', 'cec10-2014']:
                m1_cnt = np.ones(dim)

            elif func_name in ['cec13-2014']:
                m1_cnt = np.sum(m1_flag*3, axis=0)

            elif func_name in ['cec14-2014']:
                m1_cnt = np.sum(m1_flag*4, axis=0)

            elif func_name in ['cec17-2014']:
                m1_flag = m1_flag[shuffle_ay]
                pt1 = int(np.ceil(dim*0.3))
                pt2 = pt1 + int(np.ceil(dim*0.3))
                c1 = np.sum(m1_flag[:pt1], axis=0)
                c2 = np.sum(m1_flag[pt1:pt2]*2, axis=0)
                c3 = np.sum(m1_flag[pt2:], axis=0)
                m1_cnt = c1 + c2 + c3
            elif func_name in ['cec18-2014']:
                m1_flag = m1_flag[shuffle_ay]
                pt1 = int(np.ceil(dim*0.3))
                pt2 = pt1 + int(np.ceil(dim*0.3))
                c1 = np.sum(m1_flag[:pt1], axis=0)
                c2 = np.sum(m1_flag[pt1:pt2]*4, axis=0)
                c3 = np.sum(m1_flag[pt2:]*2, axis=0)
                m1_cnt = c1 + c2 + c3

            elif func_name in ['cec19-2014']:
                m1_flag = m1_flag[shuffle_ay]
                pt1 = int(np.ceil(dim*0.2))
                pt2 = pt1 + int(np.ceil(dim*0.2))
                pt3 = pt2 + int(np.ceil(dim*0.3))

                c1 = np.sum(m1_flag[:pt1]*2, axis=0)
                c2 = np.sum(m1_flag[pt1:pt2], axis=0)
                
                m2_flag = m1_flag[pt2:pt3]
                c31 = [m2_flag[0] * 2]
                c32 = m2_flag[1:-1] * 3
                c32 = np.reshape(c32, (len(m2_flag)-2, dim))
                c33 = [m2_flag[-1] * 1]
                c34 = np.concatenate((c31, c32, c33), axis=0)
                c3 = np.sum(c34, axis=0)
                c4 = np.sum(m1_flag[pt3:]*2, axis=0)
                m1_cnt = c1 + c2 + c3 + c4

            elif func_name in ['cec20-2014']:
                m1_flag = m1_flag[shuffle_ay]
                pt1 = int(np.ceil(dim*0.2))
                pt2 = pt1 + int(np.ceil(dim*0.2))
                pt3 = pt2 + int(np.ceil(dim*0.3))

                c1 = np.sum(m1_flag[:pt1]*4, axis=0)
                c2 = np.sum(m1_flag[pt1:pt2], axis=0)
                c3 = np.sum(m1_flag[pt2:pt3]*2, axis=0)
                c4 = np.sum(m1_flag[pt3:]*2, axis=0)
                m1_cnt = c1 + c2 + c3 + c4

            elif func_name in ['cec21-2014']:
                m1_flag = m1_flag[shuffle_ay]
                pt1 = int(np.ceil(dim*0.1))
                pt2 = pt1 + int(np.ceil(dim*0.2))
                pt3 = pt2 + int(np.ceil(dim*0.2))
                pt4 = pt3 + int(np.ceil(dim*0.2))

                c1 = np.sum(m1_flag[:pt1]*2, axis=0)
                c2 = np.sum(m1_flag[pt1:pt2]*4, axis=0)
                
                m2_flag = m1_flag[pt2:pt3]
                c31 = [m2_flag[0] * 2]
                c32 = m2_flag[1:-1] * 3
                c32 = np.reshape(c32, (len(m2_flag)-2, dim))
                c33 = [m2_flag[-1] * 1]
                c34 = np.concatenate((c31, c32, c33), axis=0)
                
                c3 = np.sum(c34, axis=0)
                c4 = np.sum(m1_flag[pt3:pt4], axis=0)
                c5 = np.sum(m1_flag[pt4:], axis=0)
                m1_cnt = c1 + c2 + c3 + c4 + c5
            elif func_name in ['cec22-2014']:
                m1_flag = m1_flag[shuffle_ay]
                pt1 = int(np.ceil(dim*0.1))
                pt2 = pt1 + int(np.ceil(dim*0.2))
                pt3 = pt2 + int(np.ceil(dim*0.2))
                pt4 = pt3 + int(np.ceil(dim*0.2))
                c1 = np.sum(m1_flag[:pt1]*2, axis=0)
                c2 = np.sum(m1_flag[pt1:pt2]*3, axis=0)
                c3 = np.sum(m1_flag[pt2:pt3]*2, axis=0)
                c4 = np.sum(m1_flag[pt3:pt4], axis=0)
                c5 = np.sum(m1_flag[pt4:]*2, axis=0)
                m1_cnt = c1 + c2 + c3 + c4 + c5

            else:
                raise ValueError("The func_name [{}] is not implemented..".format(func_name))

        norm_m1_cnt = m1_cnt/np.sum(m1_cnt)
        # m_cnt[func_name] = norm_m1_cnt
        disstr = ''
        for dstr in norm_m1_cnt:
            disstr = disstr + '{}, '.format(dstr)
        disstr = disstr[:-2]
        print("\"{}\": [{}],".format(func_name, disstr))



def main():
    calVarDegree()

if __name__ == '__main__':
    main()




"""
https://bee22.com/resources/Liang%20CEC2014.pdf
https://github.com/P-N-Suganthan/CEC2014/blob/master/cec14-matlab-code.zip

variale degree

******************************************************
10-Dim:
"cec1-2014": [0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.11764705882352941, 0.11764705882352941, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706],
"cec2-2014": [0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.11764705882352941],
"cec3-2014": [0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.11764705882352941, 0.11764705882352941, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706],
"cec4-2014": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
"cec5-2014": [0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.11764705882352941, 0.11764705882352941, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706],
"cec6-2014": [0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.11764705882352941],
"cec7-2014": [0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.11764705882352941, 0.11764705882352941],
"cec8-2014": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
"cec9-2014": [0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.11764705882352941, 0.11764705882352941, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706],
"cec10-2014": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
"cec11-2014": [0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.11764705882352941, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941],
"cec12-2014": [0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.11764705882352941, 0.11764705882352941, 0.08823529411764706],
"cec13-2014": [0.11764705882352941, 0.11764705882352941, 0.08823529411764706, 0.11764705882352941, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706],
"cec14-2014": [0.11764705882352941, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.11764705882352941],
"cec15-2014": [0.08823529411764706, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941, 0.11764705882352941, 0.11764705882352941],
"cec16-2014": [0.11764705882352941, 0.08823529411764706, 0.11764705882352941, 0.11764705882352941, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.08823529411764706, 0.11764705882352941],
"cec17-2014": [0.09302325581395349, 0.09302325581395349, 0.13953488372093023, 0.09302325581395349, 0.06976744186046512, 0.13953488372093023, 0.06976744186046512, 0.06976744186046512, 0.09302325581395349, 0.13953488372093023],
"cec18-2014": [0.1038961038961039, 0.15584415584415584, 0.15584415584415584, 0.03896103896103896, 0.15584415584415584, 0.1038961038961039, 0.03896103896103896, 0.1038961038961039, 0.03896103896103896, 0.1038961038961039],
"cec19-2014": [0.041666666666666664, 0.125, 0.125, 0.125, 0.041666666666666664, 0.125, 0.125, 0.08333333333333333, 0.08333333333333333, 0.125],
"cec20-2014": [0.03571428571428571, 0.10714285714285714, 0.10714285714285714, 0.10714285714285714, 0.03571428571428571, 0.10714285714285714, 0.10714285714285714, 0.10714285714285714, 0.14285714285714285, 0.14285714285714285],
"cec21-2014": [0.05405405405405406, 0.08108108108108109, 0.05405405405405406, 0.21621621621621623, 0.05405405405405406, 0.08108108108108109, 0.08108108108108109, 0.08108108108108109, 0.21621621621621623, 0.08108108108108109],
"cec22-2014": [0.09090909090909091, 0.13636363636363635, 0.045454545454545456, 0.13636363636363635, 0.09090909090909091, 0.13636363636363635, 0.045454545454545456, 0.13636363636363635, 0.13636363636363635, 0.045454545454545456],
"cec23-2014": [0.10526315789473684, 0.09868421052631579, 0.09868421052631579, 0.09539473684210527, 0.1118421052631579, 0.09868421052631579, 0.10197368421052631, 0.08881578947368421, 0.09539473684210527, 0.10526315789473684],
"cec24-2014": [0.10248447204968944, 0.11490683229813664, 0.11490683229813664, 0.07763975155279502, 0.09006211180124224, 0.10248447204968944, 0.07763975155279502, 0.11490683229813664, 0.11490683229813664, 0.09006211180124224],
"cec25-2014": [0.10576923076923077, 0.11538461538461539, 0.10576923076923077, 0.08653846153846154, 0.08653846153846154, 0.10576923076923077, 0.11538461538461539, 0.08653846153846154, 0.10576923076923077, 0.08653846153846154],
"cec26-2014": [0.10096153846153846, 0.09615384615384616, 0.09134615384615384, 0.10096153846153846, 0.11538461538461539, 0.11057692307692307, 0.09615384615384616, 0.09615384615384616, 0.10576923076923077, 0.08653846153846154],
"cec27-2014": [0.08974358974358974, 0.1111111111111111, 0.0811965811965812, 0.11538461538461539, 0.08547008547008547, 0.11538461538461539, 0.10256410256410256, 0.10256410256410256, 0.08547008547008547, 0.1111111111111111],
"cec28-2014": [0.1111111111111111, 0.09401709401709402, 0.08974358974358974, 0.09401709401709402, 0.11538461538461539, 0.08974358974358974, 0.09401709401709402, 0.09401709401709402, 0.1111111111111111, 0.10683760683760683],
"cec29-2014": [0.0979020979020979, 0.12237762237762238, 0.1048951048951049, 0.0979020979020979, 0.08391608391608392, 0.10839160839160839, 0.09090909090909091, 0.06643356643356643, 0.1048951048951049, 0.12237762237762238],
"cec30-2014": [0.07763975155279502, 0.10869565217391304, 0.12732919254658384, 0.12732919254658384, 0.07763975155279502, 0.07763975155279502, 0.10869565217391304, 0.12732919254658384, 0.07763975155279502, 0.09006211180124224],


***********************************
30-Dim:
"cec1-2014": [0.010869565217391304, 0.04891304347826087, 0.010869565217391304, 0.03804347826086957, 0.04891304347826087, 0.03804347826086957, 0.03804347826086957, 0.02717391304347826, 0.02717391304347826, 0.04891304347826087, 0.016304347826086956, 0.016304347826086956, 0.021739130434782608, 0.04891304347826087, 0.03804347826086957, 0.02717391304347826, 0.02717391304347826, 0.021739130434782608, 0.04891304347826087, 0.016304347826086956, 0.03804347826086957, 0.04891304347826087, 0.04891304347826087, 0.03804347826086957, 0.021739130434782608, 0.021739130434782608, 0.03804347826086957, 0.02717391304347826, 0.04891304347826087, 0.04891304347826087],
"cec2-2014": [0.010869565217391304, 0.04891304347826087, 0.02717391304347826, 0.04891304347826087, 0.021739130434782608, 0.03804347826086957, 0.02717391304347826, 0.04891304347826087, 0.016304347826086956, 0.03804347826086957, 0.04891304347826087, 0.02717391304347826, 0.021739130434782608, 0.04891304347826087, 0.03804347826086957, 0.016304347826086956, 0.02717391304347826, 0.021739130434782608, 0.04891304347826087, 0.04891304347826087, 0.021739130434782608, 0.03804347826086957, 0.010869565217391304, 0.016304347826086956, 0.04891304347826087, 0.04891304347826087, 0.03804347826086957, 0.02717391304347826, 0.03804347826086957, 0.03804347826086957],
"cec3-2014": [0.016304347826086956, 0.021739130434782608, 0.03804347826086957, 0.010869565217391304, 0.02717391304347826, 0.021739130434782608, 0.04891304347826087, 0.03804347826086957, 0.04891304347826087, 0.04891304347826087, 0.02717391304347826, 0.04891304347826087, 0.04891304347826087, 0.02717391304347826, 0.016304347826086956, 0.04891304347826087, 0.021739130434782608, 0.03804347826086957, 0.04891304347826087, 0.04891304347826087, 0.02717391304347826, 0.04891304347826087, 0.010869565217391304, 0.03804347826086957, 0.021739130434782608, 0.02717391304347826, 0.03804347826086957, 0.03804347826086957, 0.03804347826086957, 0.016304347826086956],
"cec4-2014": [0.03795066413662239, 0.022770398481973434, 0.04743833017077799, 0.04743833017077799, 0.028462998102466792, 0.028462998102466792, 0.022770398481973434, 0.03795066413662239, 0.04743833017077799, 0.04743833017077799, 0.011385199240986717, 0.028462998102466792, 0.03795066413662239, 0.04743833017077799, 0.04743833017077799, 0.03795066413662239, 0.04743833017077799, 0.017077798861480076, 0.028462998102466792, 0.011385199240986717, 0.03795066413662239, 0.017077798861480076, 0.022770398481973434, 0.04743833017077799, 0.03795066413662239, 0.022770398481973434, 0.017077798861480076, 0.03795066413662239, 0.028462998102466792, 0.04743833017077799],
"cec5-2014": [0.04891304347826087, 0.016304347826086956, 0.04891304347826087, 0.03804347826086957, 0.04891304347826087, 0.03804347826086957, 0.02717391304347826, 0.04891304347826087, 0.04891304347826087, 0.04891304347826087, 0.03804347826086957, 0.021739130434782608, 0.03804347826086957, 0.021739130434782608, 0.04891304347826087, 0.03804347826086957, 0.010869565217391304, 0.02717391304347826, 0.04891304347826087, 0.02717391304347826, 0.04891304347826087, 0.010869565217391304, 0.021739130434782608, 0.021739130434782608, 0.016304347826086956, 0.016304347826086956, 0.02717391304347826, 0.02717391304347826, 0.03804347826086957, 0.03804347826086957],
"cec6-2014": [0.02717391304347826, 0.02717391304347826, 0.010869565217391304, 0.04891304347826087, 0.03804347826086957, 0.03804347826086957, 0.04891304347826087, 0.04891304347826087, 0.04891304347826087, 0.02717391304347826, 0.021739130434782608, 0.03804347826086957, 0.010869565217391304, 0.04891304347826087, 0.03804347826086957, 0.021739130434782608, 0.021739130434782608, 0.04891304347826087, 0.04891304347826087, 0.016304347826086956, 0.04891304347826087, 0.016304347826086956, 0.021739130434782608, 0.04891304347826087, 0.016304347826086956, 0.03804347826086957, 0.02717391304347826, 0.03804347826086957, 0.03804347826086957, 0.02717391304347826],
"cec7-2014": [0.021739130434782608, 0.016304347826086956, 0.04891304347826087, 0.04891304347826087, 0.021739130434782608, 0.021739130434782608, 0.02717391304347826, 0.02717391304347826, 0.02717391304347826, 0.04891304347826087, 0.04891304347826087, 0.03804347826086957, 0.04891304347826087, 0.010869565217391304, 0.04891304347826087, 0.016304347826086956, 0.03804347826086957, 0.03804347826086957, 0.03804347826086957, 0.010869565217391304, 0.021739130434782608, 0.03804347826086957, 0.016304347826086956, 0.02717391304347826, 0.03804347826086957, 0.04891304347826087, 0.04891304347826087, 0.02717391304347826, 0.03804347826086957, 0.04891304347826087],
"cec8-2014": [0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333],
"cec9-2014": [0.04891304347826087, 0.010869565217391304, 0.04891304347826087, 0.04891304347826087, 0.02717391304347826, 0.02717391304347826, 0.03804347826086957, 0.016304347826086956, 0.016304347826086956, 0.021739130434782608, 0.04891304347826087, 0.04891304347826087, 0.04891304347826087, 0.03804347826086957, 0.02717391304347826, 0.03804347826086957, 0.04891304347826087, 0.02717391304347826, 0.02717391304347826, 0.03804347826086957, 0.03804347826086957, 0.016304347826086956, 0.04891304347826087, 0.010869565217391304, 0.03804347826086957, 0.04891304347826087, 0.021739130434782608, 0.03804347826086957, 0.021739130434782608, 0.021739130434782608],
"cec10-2014": [0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333],
"cec11-2014": [0.03804347826086957, 0.016304347826086956, 0.04891304347826087, 0.04891304347826087, 0.02717391304347826, 0.021739130434782608, 0.010869565217391304, 0.021739130434782608, 0.016304347826086956, 0.02717391304347826, 0.04891304347826087, 0.04891304347826087, 0.03804347826086957, 0.021739130434782608, 0.03804347826086957, 0.04891304347826087, 0.03804347826086957, 0.03804347826086957, 0.02717391304347826, 0.02717391304347826, 0.010869565217391304, 0.04891304347826087, 0.016304347826086956, 0.04891304347826087, 0.03804347826086957, 0.04891304347826087, 0.04891304347826087, 0.02717391304347826, 0.03804347826086957, 0.021739130434782608],
"cec12-2014": [0.03804347826086957, 0.02717391304347826, 0.04891304347826087, 0.02717391304347826, 0.016304347826086956, 0.04891304347826087, 0.03804347826086957, 0.02717391304347826, 0.010869565217391304, 0.03804347826086957, 0.04891304347826087, 0.03804347826086957, 0.03804347826086957, 0.04891304347826087, 0.010869565217391304, 0.04891304347826087, 0.021739130434782608, 0.04891304347826087, 0.02717391304347826, 0.021739130434782608, 0.04891304347826087, 0.021739130434782608, 0.03804347826086957, 0.04891304347826087, 0.03804347826086957, 0.04891304347826087, 0.021739130434782608, 0.016304347826086956, 0.016304347826086956, 0.02717391304347826],
"cec13-2014": [0.03804347826086957, 0.021739130434782608, 0.04891304347826087, 0.016304347826086956, 0.021739130434782608, 0.016304347826086956, 0.03804347826086957, 0.04891304347826087, 0.03804347826086957, 0.03804347826086957, 0.04891304347826087, 0.016304347826086956, 0.04891304347826087, 0.021739130434782608, 0.02717391304347826, 0.021739130434782608, 0.02717391304347826, 0.04891304347826087, 0.02717391304347826, 0.02717391304347826, 0.04891304347826087, 0.04891304347826087, 0.03804347826086957, 0.04891304347826087, 0.010869565217391304, 0.04891304347826087, 0.03804347826086957, 0.03804347826086957, 0.010869565217391304, 0.02717391304347826],
"cec14-2014": [0.021739130434782608, 0.02717391304347826, 0.016304347826086956, 0.03804347826086957, 0.03804347826086957, 0.03804347826086957, 0.016304347826086956, 0.04891304347826087, 0.04891304347826087, 0.02717391304347826, 0.02717391304347826, 0.021739130434782608, 0.021739130434782608, 0.03804347826086957, 0.04891304347826087, 0.04891304347826087, 0.04891304347826087, 0.03804347826086957, 0.03804347826086957, 0.010869565217391304, 0.04891304347826087, 0.010869565217391304, 0.03804347826086957, 0.016304347826086956, 0.04891304347826087, 0.04891304347826087, 0.021739130434782608, 0.02717391304347826, 0.04891304347826087, 0.02717391304347826],
"cec15-2014": [0.03804347826086957, 0.04891304347826087, 0.021739130434782608, 0.016304347826086956, 0.03804347826086957, 0.016304347826086956, 0.02717391304347826, 0.010869565217391304, 0.02717391304347826, 0.010869565217391304, 0.021739130434782608, 0.03804347826086957, 0.03804347826086957, 0.04891304347826087, 0.02717391304347826, 0.02717391304347826, 0.04891304347826087, 0.02717391304347826, 0.04891304347826087, 0.04891304347826087, 0.021739130434782608, 0.016304347826086956, 0.04891304347826087, 0.03804347826086957, 0.04891304347826087, 0.021739130434782608, 0.03804347826086957, 0.04891304347826087, 0.04891304347826087, 0.03804347826086957],
"cec16-2014": [0.02717391304347826, 0.03804347826086957, 0.04891304347826087, 0.03804347826086957, 0.016304347826086956, 0.02717391304347826, 0.04891304347826087, 0.04891304347826087, 0.03804347826086957, 0.021739130434782608, 0.010869565217391304, 0.02717391304347826, 0.02717391304347826, 0.021739130434782608, 0.03804347826086957, 0.04891304347826087, 0.016304347826086956, 0.021739130434782608, 0.021739130434782608, 0.04891304347826087, 0.04891304347826087, 0.04891304347826087, 0.04891304347826087, 0.03804347826086957, 0.010869565217391304, 0.016304347826086956, 0.04891304347826087, 0.03804347826086957, 0.02717391304347826, 0.03804347826086957],
"cec17-2014": [0.023255813953488372, 0.023255813953488372, 0.031007751937984496, 0.023255813953488372, 0.031007751937984496, 0.046511627906976744, 0.031007751937984496, 0.046511627906976744, 0.031007751937984496, 0.023255813953488372, 0.046511627906976744, 0.031007751937984496, 0.046511627906976744, 0.023255813953488372, 0.046511627906976744, 0.031007751937984496, 0.031007751937984496, 0.031007751937984496, 0.023255813953488372, 0.023255813953488372, 0.023255813953488372, 0.046511627906976744, 0.031007751937984496, 0.046511627906976744, 0.046511627906976744, 0.031007751937984496, 0.023255813953488372, 0.046511627906976744, 0.031007751937984496, 0.031007751937984496],
"cec18-2014": [0.03463203463203463, 0.03463203463203463, 0.03463203463203463, 0.05194805194805195, 0.05194805194805195, 0.012987012987012988, 0.012987012987012988, 0.05194805194805195, 0.03463203463203463, 0.05194805194805195, 0.05194805194805195, 0.03463203463203463, 0.03463203463203463, 0.03463203463203463, 0.03463203463203463, 0.03463203463203463, 0.012987012987012988, 0.05194805194805195, 0.012987012987012988, 0.03463203463203463, 0.05194805194805195, 0.012987012987012988, 0.012987012987012988, 0.03463203463203463, 0.05194805194805195, 0.012987012987012988, 0.012987012987012988, 0.012987012987012988, 0.03463203463203463, 0.05194805194805195],
"cec19-2014": [0.04938271604938271, 0.037037037037037035, 0.024691358024691357, 0.04938271604938271, 0.037037037037037035, 0.037037037037037035, 0.024691358024691357, 0.04938271604938271, 0.024691358024691357, 0.04938271604938271, 0.012345679012345678, 0.012345679012345678, 0.04938271604938271, 0.04938271604938271, 0.024691358024691357, 0.012345679012345678, 0.037037037037037035, 0.037037037037037035, 0.04938271604938271, 0.037037037037037035, 0.04938271604938271, 0.024691358024691357, 0.012345679012345678, 0.012345679012345678, 0.037037037037037035, 0.037037037037037035, 0.04938271604938271, 0.024691358024691357, 0.012345679012345678, 0.037037037037037035],
"cec20-2014": [0.047619047619047616, 0.011904761904761904, 0.03571428571428571, 0.047619047619047616, 0.047619047619047616, 0.03571428571428571, 0.03571428571428571, 0.03571428571428571, 0.03571428571428571, 0.03571428571428571, 0.03571428571428571, 0.047619047619047616, 0.03571428571428571, 0.03571428571428571, 0.011904761904761904, 0.03571428571428571, 0.03571428571428571, 0.03571428571428571, 0.011904761904761904, 0.011904761904761904, 0.011904761904761904, 0.03571428571428571, 0.047619047619047616, 0.047619047619047616, 0.03571428571428571, 0.03571428571428571, 0.03571428571428571, 0.03571428571428571, 0.03571428571428571, 0.011904761904761904],
"cec21-2014": [0.024390243902439025, 0.04065040650406504, 0.06504065040650407, 0.024390243902439025, 0.04065040650406504, 0.04065040650406504, 0.024390243902439025, 0.024390243902439025, 0.06504065040650407, 0.016260162601626018, 0.024390243902439025, 0.016260162601626018, 0.016260162601626018, 0.024390243902439025, 0.024390243902439025, 0.016260162601626018, 0.04065040650406504, 0.016260162601626018, 0.06504065040650407, 0.06504065040650407, 0.06504065040650407, 0.016260162601626018, 0.04065040650406504, 0.016260162601626018, 0.016260162601626018, 0.06504065040650407, 0.04065040650406504, 0.024390243902439025, 0.016260162601626018, 0.024390243902439025],
"cec22-2014": [0.015151515151515152, 0.045454545454545456, 0.015151515151515152, 0.015151515151515152, 0.030303030303030304, 0.045454545454545456, 0.030303030303030304, 0.045454545454545456, 0.015151515151515152, 0.015151515151515152, 0.045454545454545456, 0.015151515151515152, 0.015151515151515152, 0.045454545454545456, 0.045454545454545456, 0.045454545454545456, 0.030303030303030304, 0.045454545454545456, 0.045454545454545456, 0.045454545454545456, 0.015151515151515152, 0.045454545454545456, 0.045454545454545456, 0.045454545454545456, 0.015151515151515152, 0.045454545454545456, 0.030303030303030304, 0.045454545454545456, 0.030303030303030304, 0.030303030303030304],
"cec23-2014": [0.03739837398373984, 0.03333333333333333, 0.034959349593495934, 0.036585365853658534, 0.03414634146341464, 0.03333333333333333, 0.030894308943089432, 0.034959349593495934, 0.03739837398373984, 0.03739837398373984, 0.032520325203252036, 0.028455284552845527, 0.03170731707317073, 0.02032520325203252, 0.03333333333333333, 0.03902439024390244, 0.034959349593495934, 0.03333333333333333, 0.03008130081300813, 0.034959349593495934, 0.03333333333333333, 0.03414634146341464, 0.036585365853658534, 0.034959349593495934, 0.036585365853658534, 0.01869918699186992, 0.03333333333333333, 0.03333333333333333, 0.03170731707317073, 0.038211382113821135],
"cec24-2014": [0.019936204146730464, 0.03907496012759171, 0.03269537480063796, 0.03907496012759171, 0.019936204146730464, 0.03588516746411483, 0.029505582137161084, 0.03907496012759171, 0.03588516746411483, 0.03907496012759171, 0.03907496012759171, 0.029505582137161084, 0.023125996810207338, 0.03588516746411483, 0.03269537480063796, 0.029505582137161084, 0.029505582137161084, 0.029505582137161084, 0.029505582137161084, 0.03907496012759171, 0.03907496012759171, 0.03269537480063796, 0.03269537480063796, 0.03907496012759171, 0.03588516746411483, 0.03269537480063796, 0.03588516746411483, 0.03588516746411483, 0.029505582137161084, 0.03907496012759171],
"cec25-2014": [0.031862745098039214, 0.031862745098039214, 0.03676470588235294, 0.022058823529411766, 0.03676470588235294, 0.02696078431372549, 0.03431372549019608, 0.03431372549019608, 0.03676470588235294, 0.0392156862745098, 0.0392156862745098, 0.031862745098039214, 0.03431372549019608, 0.0392156862745098, 0.0392156862745098, 0.024509803921568627, 0.031862745098039214, 0.03431372549019608, 0.03676470588235294, 0.029411764705882353, 0.03431372549019608, 0.031862745098039214, 0.03676470588235294, 0.024509803921568627, 0.031862745098039214, 0.029411764705882353, 0.031862745098039214, 0.031862745098039214, 0.0392156862745098, 0.03676470588235294],
"cec26-2014": [0.03799019607843137, 0.03431372549019608, 0.030637254901960783, 0.03553921568627451, 0.03799019607843137, 0.03431372549019608, 0.03553921568627451, 0.03308823529411765, 0.01838235294117647, 0.03431372549019608, 0.029411764705882353, 0.03676470588235294, 0.03308823529411765, 0.03799019607843137, 0.030637254901960783, 0.031862745098039214, 0.031862745098039214, 0.03308823529411765, 0.03553921568627451, 0.03799019607843137, 0.03431372549019608, 0.03676470588235294, 0.0392156862745098, 0.023284313725490197, 0.023284313725490197, 0.03308823529411765, 0.03799019607843137, 0.030637254901960783, 0.03676470588235294, 0.03431372549019608],
"cec27-2014": [0.03594771241830065, 0.030501089324618737, 0.03376906318082789, 0.034858387799564274, 0.030501089324618737, 0.0392156862745098, 0.03376906318082789, 0.02287581699346405, 0.03376906318082789, 0.032679738562091505, 0.030501089324618737, 0.03812636165577342, 0.03376906318082789, 0.030501089324618737, 0.034858387799564274, 0.037037037037037035, 0.032679738562091505, 0.02178649237472767, 0.03594771241830065, 0.034858387799564274, 0.0392156862745098, 0.03812636165577342, 0.029411764705882353, 0.037037037037037035, 0.027233115468409588, 0.03594771241830065, 0.030501089324618737, 0.034858387799564274, 0.03159041394335512, 0.03812636165577342],
"cec28-2014": [0.03159041394335512, 0.0392156862745098, 0.03594771241830065, 0.027233115468409588, 0.02832244008714597, 0.03594771241830065, 0.034858387799564274, 0.030501089324618737, 0.030501089324618737, 0.032679738562091505, 0.029411764705882353, 0.03812636165577342, 0.03376906318082789, 0.0392156862745098, 0.03159041394335512, 0.026143790849673203, 0.034858387799564274, 0.03812636165577342, 0.03594771241830065, 0.03159041394335512, 0.037037037037037035, 0.03376906318082789, 0.026143790849673203, 0.03376906318082789, 0.03812636165577342, 0.037037037037037035, 0.03812636165577342, 0.026143790849673203, 0.03376906318082789, 0.030501089324618737],
"cec29-2014": [0.05102040816326531, 0.03741496598639456, 0.030612244897959183, 0.036564625850340135, 0.04421768707482993, 0.031462585034013606, 0.03316326530612245, 0.025510204081632654, 0.022108843537414966, 0.03231292517006803, 0.02465986394557823, 0.030612244897959183, 0.02806122448979592, 0.023809523809523808, 0.02040816326530612, 0.03486394557823129, 0.031462585034013606, 0.030612244897959183, 0.04421768707482993, 0.034013605442176874, 0.050170068027210885, 0.0391156462585034, 0.013605442176870748, 0.03741496598639456, 0.034013605442176874, 0.04336734693877551, 0.03231292517006803, 0.026360544217687076, 0.031462585034013606, 0.045068027210884355],
"cec30-2014": [0.030032467532467532, 0.026785714285714284, 0.0349025974025974, 0.03165584415584415, 0.026785714285714284, 0.020292207792207792, 0.030844155844155844, 0.03814935064935065, 0.032467532467532464, 0.030032467532467532, 0.03165584415584415, 0.03165584415584415, 0.03814935064935065, 0.03977272727272727, 0.02353896103896104, 0.030032467532467532, 0.036525974025974024, 0.04626623376623377, 0.041396103896103896, 0.03165584415584415, 0.0349025974025974, 0.0349025974025974, 0.041396103896103896, 0.041396103896103896, 0.03165584415584415, 0.03165584415584415, 0.0349025974025974, 0.0349025974025974, 0.03814935064935065, 0.02353896103896104],

***********************************
50-Dim:
"cec1-2014": [0.02857142857142857, 0.02857142857142857, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.011428571428571429, 0.022857142857142857, 0.014285714285714285, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.011428571428571429, 0.017142857142857144, 0.008571428571428572, 0.017142857142857144, 0.02857142857142857, 0.014285714285714285, 0.022857142857142857, 0.017142857142857144, 0.02857142857142857, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.017142857142857144, 0.017142857142857144, 0.02857142857142857, 0.014285714285714285, 0.022857142857142857, 0.008571428571428572, 0.017142857142857144, 0.017142857142857144, 0.011428571428571429, 0.014285714285714285, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.011428571428571429, 0.008571428571428572, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144],
"cec2-2014": [0.02857142857142857, 0.017142857142857144, 0.014285714285714285, 0.014285714285714285, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.017142857142857144, 0.011428571428571429, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.008571428571428572, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.011428571428571429, 0.017142857142857144, 0.017142857142857144, 0.02857142857142857, 0.008571428571428572, 0.02857142857142857, 0.02857142857142857, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.008571428571428572, 0.02857142857142857, 0.014285714285714285, 0.017142857142857144, 0.022857142857142857, 0.011428571428571429, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.011428571428571429, 0.017142857142857144, 0.017142857142857144],
"cec3-2014": [0.022857142857142857, 0.011428571428571429, 0.017142857142857144, 0.02857142857142857, 0.008571428571428572, 0.02857142857142857, 0.02857142857142857, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.011428571428571429, 0.014285714285714285, 0.011428571428571429, 0.014285714285714285, 0.017142857142857144, 0.017142857142857144, 0.011428571428571429, 0.02857142857142857, 0.017142857142857144, 0.008571428571428572, 0.02857142857142857, 0.022857142857142857, 0.008571428571428572, 0.02857142857142857, 0.022857142857142857, 0.02857142857142857, 0.014285714285714285, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.014285714285714285, 0.017142857142857144, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.02857142857142857, 0.014285714285714285],
"cec4-2014": [0.016536964980544747, 0.021400778210116732, 0.014591439688715954, 0.021400778210116732, 0.023346303501945526, 0.014591439688715954, 0.029182879377431907, 0.008754863813229572, 0.021400778210116732, 0.029182879377431907, 0.023346303501945526, 0.011673151750972763, 0.016536964980544747, 0.017509727626459144, 0.029182879377431907, 0.023346303501945526, 0.023346303501945526, 0.017509727626459144, 0.023346303501945526, 0.014591439688715954, 0.016536964980544747, 0.008754863813229572, 0.014591439688715954, 0.017509727626459144, 0.029182879377431907, 0.021400778210116732, 0.029182879377431907, 0.029182879377431907, 0.017509727626459144, 0.023346303501945526, 0.011673151750972763, 0.023346303501945526, 0.008754863813229572, 0.017509727626459144, 0.011673151750972763, 0.021400778210116732, 0.021400778210116732, 0.016536964980544747, 0.029182879377431907, 0.021400778210116732, 0.014591439688715954, 0.023346303501945526, 0.011673151750972763, 0.016536964980544747, 0.029182879377431907, 0.016536964980544747, 0.017509727626459144, 0.029182879377431907, 0.029182879377431907, 0.021400778210116732],
"cec5-2014": [0.02857142857142857, 0.011428571428571429, 0.017142857142857144, 0.008571428571428572, 0.017142857142857144, 0.017142857142857144, 0.02857142857142857, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.017142857142857144, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.011428571428571429, 0.017142857142857144, 0.011428571428571429, 0.02857142857142857, 0.022857142857142857, 0.02857142857142857, 0.02857142857142857, 0.011428571428571429, 0.017142857142857144, 0.02857142857142857, 0.017142857142857144, 0.017142857142857144, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.017142857142857144, 0.022857142857142857, 0.02857142857142857, 0.008571428571428572, 0.022857142857142857, 0.008571428571428572, 0.014285714285714285, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.022857142857142857, 0.017142857142857144],
"cec6-2014": [0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.02857142857142857, 0.022857142857142857, 0.014285714285714285, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.02857142857142857, 0.022857142857142857, 0.011428571428571429, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.014285714285714285, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.011428571428571429, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.017142857142857144, 0.011428571428571429, 0.02857142857142857, 0.022857142857142857, 0.014285714285714285, 0.011428571428571429, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.008571428571428572, 0.02857142857142857, 0.02857142857142857, 0.017142857142857144, 0.017142857142857144, 0.008571428571428572, 0.017142857142857144, 0.02857142857142857, 0.008571428571428572, 0.017142857142857144, 0.017142857142857144, 0.014285714285714285, 0.022857142857142857, 0.017142857142857144],
"cec7-2014": [0.02857142857142857, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.02857142857142857, 0.008571428571428572, 0.017142857142857144, 0.008571428571428572, 0.022857142857142857, 0.017142857142857144, 0.02857142857142857, 0.017142857142857144, 0.014285714285714285, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.014285714285714285, 0.011428571428571429, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.011428571428571429, 0.022857142857142857, 0.014285714285714285, 0.011428571428571429, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.017142857142857144, 0.02857142857142857, 0.017142857142857144, 0.02857142857142857, 0.014285714285714285, 0.017142857142857144, 0.022857142857142857, 0.011428571428571429, 0.008571428571428572, 0.017142857142857144, 0.017142857142857144, 0.02857142857142857, 0.017142857142857144, 0.02857142857142857],
"cec8-2014": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
"cec9-2014": [0.017142857142857144, 0.02857142857142857, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.02857142857142857, 0.02857142857142857, 0.014285714285714285, 0.022857142857142857, 0.011428571428571429, 0.022857142857142857, 0.017142857142857144, 0.017142857142857144, 0.011428571428571429, 0.008571428571428572, 0.008571428571428572, 0.02857142857142857, 0.014285714285714285, 0.022857142857142857, 0.014285714285714285, 0.02857142857142857, 0.022857142857142857, 0.014285714285714285, 0.02857142857142857, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.011428571428571429, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.014285714285714285, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.008571428571428572, 0.011428571428571429, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.017142857142857144, 0.017142857142857144, 0.02857142857142857],
"cec10-2014": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
"cec11-2014": [0.022857142857142857, 0.017142857142857144, 0.008571428571428572, 0.022857142857142857, 0.022857142857142857, 0.011428571428571429, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.02857142857142857, 0.02857142857142857, 0.008571428571428572, 0.008571428571428572, 0.017142857142857144, 0.017142857142857144, 0.017142857142857144, 0.017142857142857144, 0.017142857142857144, 0.017142857142857144, 0.014285714285714285, 0.02857142857142857, 0.014285714285714285, 0.022857142857142857, 0.011428571428571429, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.014285714285714285, 0.02857142857142857, 0.022857142857142857, 0.017142857142857144, 0.02857142857142857, 0.011428571428571429, 0.022857142857142857, 0.014285714285714285, 0.02857142857142857, 0.011428571428571429, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.02857142857142857, 0.022857142857142857],
"cec12-2014": [0.017142857142857144, 0.011428571428571429, 0.008571428571428572, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.02857142857142857, 0.014285714285714285, 0.02857142857142857, 0.008571428571428572, 0.014285714285714285, 0.022857142857142857, 0.017142857142857144, 0.014285714285714285, 0.022857142857142857, 0.022857142857142857, 0.011428571428571429, 0.011428571428571429, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.008571428571428572, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.02857142857142857, 0.02857142857142857, 0.017142857142857144, 0.02857142857142857, 0.022857142857142857, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.02857142857142857, 0.017142857142857144, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.011428571428571429, 0.017142857142857144, 0.014285714285714285, 0.017142857142857144],
"cec13-2014": [0.022857142857142857, 0.011428571428571429, 0.014285714285714285, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.014285714285714285, 0.022857142857142857, 0.017142857142857144, 0.02857142857142857, 0.014285714285714285, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.011428571428571429, 0.011428571428571429, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.022857142857142857, 0.022857142857142857, 0.011428571428571429, 0.008571428571428572, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.017142857142857144, 0.02857142857142857, 0.02857142857142857, 0.017142857142857144, 0.02857142857142857, 0.02857142857142857, 0.02857142857142857, 0.02857142857142857, 0.02857142857142857, 0.022857142857142857, 0.014285714285714285, 0.008571428571428572, 0.008571428571428572, 0.017142857142857144],
"cec14-2014": [0.011428571428571429, 0.022857142857142857, 0.011428571428571429, 0.022857142857142857, 0.017142857142857144, 0.008571428571428572, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.017142857142857144, 0.02857142857142857, 0.022857142857142857, 0.017142857142857144, 0.014285714285714285, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.014285714285714285, 0.02857142857142857, 0.011428571428571429, 0.017142857142857144, 0.022857142857142857, 0.008571428571428572, 0.022857142857142857, 0.017142857142857144, 0.02857142857142857, 0.02857142857142857, 0.014285714285714285, 0.02857142857142857, 0.011428571428571429, 0.017142857142857144, 0.017142857142857144, 0.008571428571428572, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.02857142857142857, 0.014285714285714285, 0.022857142857142857, 0.014285714285714285, 0.02857142857142857],
"cec15-2014": [0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.017142857142857144, 0.02857142857142857, 0.022857142857142857, 0.008571428571428572, 0.02857142857142857, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.014285714285714285, 0.017142857142857144, 0.022857142857142857, 0.011428571428571429, 0.02857142857142857, 0.017142857142857144, 0.022857142857142857, 0.02857142857142857, 0.014285714285714285, 0.011428571428571429, 0.02857142857142857, 0.017142857142857144, 0.014285714285714285, 0.014285714285714285, 0.017142857142857144, 0.02857142857142857, 0.011428571428571429, 0.02857142857142857, 0.022857142857142857, 0.02857142857142857, 0.022857142857142857, 0.017142857142857144, 0.014285714285714285, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.008571428571428572, 0.011428571428571429, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.02857142857142857, 0.008571428571428572, 0.022857142857142857, 0.022857142857142857],
"cec16-2014": [0.017142857142857144, 0.017142857142857144, 0.017142857142857144, 0.017142857142857144, 0.022857142857142857, 0.011428571428571429, 0.008571428571428572, 0.011428571428571429, 0.022857142857142857, 0.022857142857142857, 0.022857142857142857, 0.014285714285714285, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.02857142857142857, 0.02857142857142857, 0.017142857142857144, 0.022857142857142857, 0.02857142857142857, 0.02857142857142857, 0.02857142857142857, 0.017142857142857144, 0.017142857142857144, 0.011428571428571429, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.02857142857142857, 0.017142857142857144, 0.022857142857142857, 0.008571428571428572, 0.02857142857142857, 0.014285714285714285, 0.017142857142857144, 0.022857142857142857, 0.02857142857142857, 0.014285714285714285, 0.014285714285714285, 0.022857142857142857, 0.008571428571428572, 0.014285714285714285, 0.022857142857142857, 0.022857142857142857, 0.017142857142857144, 0.022857142857142857, 0.011428571428571429, 0.02857142857142857, 0.022857142857142857],
"cec17-2014": [0.013953488372093023, 0.013953488372093023, 0.018604651162790697, 0.018604651162790697, 0.027906976744186046, 0.027906976744186046, 0.027906976744186046, 0.018604651162790697, 0.027906976744186046, 0.013953488372093023, 0.027906976744186046, 0.018604651162790697, 0.027906976744186046, 0.027906976744186046, 0.013953488372093023, 0.027906976744186046, 0.018604651162790697, 0.013953488372093023, 0.013953488372093023, 0.013953488372093023, 0.027906976744186046, 0.018604651162790697, 0.018604651162790697, 0.027906976744186046, 0.013953488372093023, 0.018604651162790697, 0.013953488372093023, 0.018604651162790697, 0.013953488372093023, 0.018604651162790697, 0.018604651162790697, 0.018604651162790697, 0.018604651162790697, 0.013953488372093023, 0.013953488372093023, 0.027906976744186046, 0.018604651162790697, 0.018604651162790697, 0.013953488372093023, 0.027906976744186046, 0.018604651162790697, 0.027906976744186046, 0.018604651162790697, 0.027906976744186046, 0.018604651162790697, 0.018604651162790697, 0.013953488372093023, 0.027906976744186046, 0.018604651162790697, 0.013953488372093023],
"cec18-2014": [0.02077922077922078, 0.03116883116883117, 0.007792207792207792, 0.02077922077922078, 0.03116883116883117, 0.02077922077922078, 0.02077922077922078, 0.02077922077922078, 0.007792207792207792, 0.007792207792207792, 0.007792207792207792, 0.02077922077922078, 0.02077922077922078, 0.007792207792207792, 0.007792207792207792, 0.02077922077922078, 0.03116883116883117, 0.02077922077922078, 0.007792207792207792, 0.03116883116883117, 0.007792207792207792, 0.02077922077922078, 0.007792207792207792, 0.03116883116883117, 0.03116883116883117, 0.02077922077922078, 0.02077922077922078, 0.03116883116883117, 0.03116883116883117, 0.03116883116883117, 0.007792207792207792, 0.007792207792207792, 0.03116883116883117, 0.02077922077922078, 0.007792207792207792, 0.03116883116883117, 0.007792207792207792, 0.03116883116883117, 0.007792207792207792, 0.03116883116883117, 0.02077922077922078, 0.03116883116883117, 0.02077922077922078, 0.02077922077922078, 0.02077922077922078, 0.02077922077922078, 0.007792207792207792, 0.02077922077922078, 0.03116883116883117, 0.02077922077922078],
"cec19-2014": [0.021739130434782608, 0.007246376811594203, 0.007246376811594203, 0.007246376811594203, 0.021739130434782608, 0.021739130434782608, 0.007246376811594203, 0.014492753623188406, 0.030434782608695653, 0.021739130434782608, 0.021739130434782608, 0.021739130434782608, 0.014492753623188406, 0.007246376811594203, 0.030434782608695653, 0.030434782608695653, 0.030434782608695653, 0.007246376811594203, 0.007246376811594203, 0.014492753623188406, 0.021739130434782608, 0.021739130434782608, 0.007246376811594203, 0.030434782608695653, 0.030434782608695653, 0.030434782608695653, 0.030434782608695653, 0.014492753623188406, 0.030434782608695653, 0.021739130434782608, 0.014492753623188406, 0.007246376811594203, 0.021739130434782608, 0.014492753623188406, 0.030434782608695653, 0.021739130434782608, 0.021739130434782608, 0.030434782608695653, 0.021739130434782608, 0.014492753623188406, 0.030434782608695653, 0.014492753623188406, 0.030434782608695653, 0.014492753623188406, 0.030434782608695653, 0.030434782608695653, 0.021739130434782608, 0.021739130434782608, 0.014492753623188406, 0.007246376811594203],
"cec20-2014": [0.007142857142857143, 0.02857142857142857, 0.02142857142857143, 0.02142857142857143, 0.02857142857142857, 0.02142857142857143, 0.02142857142857143, 0.02142857142857143, 0.02142857142857143, 0.02142857142857143, 0.02142857142857143, 0.007142857142857143, 0.007142857142857143, 0.007142857142857143, 0.007142857142857143, 0.02142857142857143, 0.02142857142857143, 0.02857142857142857, 0.02142857142857143, 0.02142857142857143, 0.007142857142857143, 0.02142857142857143, 0.02142857142857143, 0.02142857142857143, 0.02142857142857143, 0.02142857142857143, 0.007142857142857143, 0.02142857142857143, 0.02142857142857143, 0.02857142857142857, 0.02142857142857143, 0.02857142857142857, 0.02142857142857143, 0.007142857142857143, 0.02142857142857143, 0.02142857142857143, 0.02142857142857143, 0.02142857142857143, 0.007142857142857143, 0.02857142857142857, 0.02857142857142857, 0.02142857142857143, 0.02857142857142857, 0.02142857142857143, 0.02857142857142857, 0.007142857142857143, 0.02142857142857143, 0.02857142857142857, 0.02142857142857143, 0.02142857142857143],
"cec21-2014": [0.02583732057416268, 0.009569377990430622, 0.014354066985645933, 0.009569377990430622, 0.03827751196172249, 0.02583732057416268, 0.009569377990430622, 0.014354066985645933, 0.014354066985645933, 0.014354066985645933, 0.009569377990430622, 0.014354066985645933, 0.009569377990430622, 0.009569377990430622, 0.03827751196172249, 0.014354066985645933, 0.009569377990430622, 0.014354066985645933, 0.02583732057416268, 0.009569377990430622, 0.014354066985645933, 0.009569377990430622, 0.014354066985645933, 0.014354066985645933, 0.014354066985645933, 0.009569377990430622, 0.009569377990430622, 0.02583732057416268, 0.02583732057416268, 0.03827751196172249, 0.009569377990430622, 0.03827751196172249, 0.009569377990430622, 0.014354066985645933, 0.03827751196172249, 0.03827751196172249, 0.014354066985645933, 0.03827751196172249, 0.03827751196172249, 0.02583732057416268, 0.009569377990430622, 0.02583732057416268, 0.03827751196172249, 0.014354066985645933, 0.02583732057416268, 0.014354066985645933, 0.03827751196172249, 0.009569377990430622, 0.02583732057416268, 0.02583732057416268],
"cec22-2014": [0.02727272727272727, 0.00909090909090909, 0.00909090909090909, 0.00909090909090909, 0.02727272727272727, 0.02727272727272727, 0.00909090909090909, 0.02727272727272727, 0.02727272727272727, 0.01818181818181818, 0.02727272727272727, 0.02727272727272727, 0.02727272727272727, 0.02727272727272727, 0.02727272727272727, 0.02727272727272727, 0.02727272727272727, 0.01818181818181818, 0.00909090909090909, 0.01818181818181818, 0.00909090909090909, 0.02727272727272727, 0.00909090909090909, 0.00909090909090909, 0.01818181818181818, 0.02727272727272727, 0.02727272727272727, 0.00909090909090909, 0.02727272727272727, 0.01818181818181818, 0.00909090909090909, 0.01818181818181818, 0.00909090909090909, 0.02727272727272727, 0.02727272727272727, 0.00909090909090909, 0.02727272727272727, 0.01818181818181818, 0.02727272727272727, 0.00909090909090909, 0.02727272727272727, 0.00909090909090909, 0.02727272727272727, 0.02727272727272727, 0.01818181818181818, 0.00909090909090909, 0.02727272727272727, 0.01818181818181818, 0.01818181818181818, 0.02727272727272727],
"cec23-2014": [0.018806214227309895, 0.019215044971381847, 0.019215044971381847, 0.022485690923957483, 0.02330335241210139, 0.019623875715453803, 0.016762060506950123, 0.020032706459525755, 0.020032706459525755, 0.0241210139002453, 0.020032706459525755, 0.01839738348323794, 0.020032706459525755, 0.008585445625511038, 0.02044153720359771, 0.016762060506950123, 0.0241210139002453, 0.01839738348323794, 0.019215044971381847, 0.022485690923957483, 0.02044153720359771, 0.022485690923957483, 0.0241210139002453, 0.020032706459525755, 0.01839738348323794, 0.022485690923957483, 0.02330335241210139, 0.019215044971381847, 0.02330335241210139, 0.015944399018806215, 0.019215044971381847, 0.021668029435813575, 0.01839738348323794, 0.019215044971381847, 0.011038430089942763, 0.020850367947669663, 0.018806214227309895, 0.022485690923957483, 0.019623875715453803, 0.019215044971381847, 0.022485690923957483, 0.020850367947669663, 0.02044153720359771, 0.01839738348323794, 0.020032706459525755, 0.022076860179885527, 0.020032706459525755, 0.020032706459525755, 0.022485690923957483, 0.020850367947669663],
"cec24-2014": [0.02465642683912692, 0.016572352465642683, 0.019805982215036377, 0.02465642683912692, 0.010105092966855295, 0.02465642683912692, 0.019805982215036377, 0.021422797089733225, 0.02465642683912692, 0.014955537590945837, 0.02465642683912692, 0.019805982215036377, 0.014955537590945837, 0.023039611964430072, 0.02465642683912692, 0.023039611964430072, 0.01818916734033953, 0.014955537590945837, 0.021422797089733225, 0.021422797089733225, 0.019805982215036377, 0.01818916734033953, 0.01818916734033953, 0.014955537590945837, 0.01818916734033953, 0.021422797089733225, 0.02465642683912692, 0.011721907841552142, 0.02465642683912692, 0.019805982215036377, 0.02465642683912692, 0.01818916734033953, 0.01818916734033953, 0.01818916734033953, 0.023039611964430072, 0.023039611964430072, 0.02465642683912692, 0.023039611964430072, 0.021422797089733225, 0.01818916734033953, 0.016572352465642683, 0.01818916734033953, 0.021422797089733225, 0.011721907841552142, 0.021422797089733225, 0.01818916734033953, 0.023039611964430072, 0.019805982215036377, 0.016572352465642683, 0.021422797089733225],
"cec25-2014": [0.023514851485148515, 0.018564356435643563, 0.019801980198019802, 0.019801980198019802, 0.018564356435643563, 0.019801980198019802, 0.017326732673267328, 0.01485148514851485, 0.01485148514851485, 0.01485148514851485, 0.022277227722772276, 0.023514851485148515, 0.019801980198019802, 0.023514851485148515, 0.023514851485148515, 0.024752475247524754, 0.01485148514851485, 0.017326732673267328, 0.019801980198019802, 0.022277227722772276, 0.02103960396039604, 0.017326732673267328, 0.02103960396039604, 0.01485148514851485, 0.022277227722772276, 0.022277227722772276, 0.018564356435643563, 0.019801980198019802, 0.017326732673267328, 0.02103960396039604, 0.018564356435643563, 0.02103960396039604, 0.02103960396039604, 0.019801980198019802, 0.019801980198019802, 0.023514851485148515, 0.022277227722772276, 0.018564356435643563, 0.019801980198019802, 0.02103960396039604, 0.017326732673267328, 0.022277227722772276, 0.012376237623762377, 0.024752475247524754, 0.018564356435643563, 0.022277227722772276, 0.023514851485148515, 0.024752475247524754, 0.018564356435643563, 0.02103960396039604],
"cec26-2014": [0.02042079207920792, 0.022896039603960396, 0.019801980198019802, 0.01608910891089109, 0.01608910891089109, 0.023514851485148515, 0.017326732673267328, 0.024133663366336634, 0.02042079207920792, 0.019183168316831683, 0.017326732673267328, 0.019801980198019802, 0.021658415841584157, 0.021658415841584157, 0.022277227722772276, 0.023514851485148515, 0.022896039603960396, 0.022277227722772276, 0.019801980198019802, 0.021658415841584157, 0.02042079207920792, 0.019183168316831683, 0.02042079207920792, 0.02042079207920792, 0.01670792079207921, 0.02103960396039604, 0.02103960396039604, 0.023514851485148515, 0.018564356435643563, 0.01547029702970297, 0.023514851485148515, 0.019183168316831683, 0.02042079207920792, 0.018564356435643563, 0.017945544554455444, 0.01608910891089109, 0.02103960396039604, 0.017945544554455444, 0.02103960396039604, 0.019183168316831683, 0.02042079207920792, 0.023514851485148515, 0.021658415841584157, 0.02103960396039604, 0.021658415841584157, 0.014232673267326733, 0.02103960396039604, 0.012995049504950494, 0.023514851485148515, 0.01547029702970297],
"cec27-2014": [0.020902090209020903, 0.022002200220022004, 0.019801980198019802, 0.017051705170517052, 0.02145214521452145, 0.020902090209020903, 0.019251925192519254, 0.0187018701870187, 0.0176017601760176, 0.02145214521452145, 0.022002200220022004, 0.0165016501650165, 0.0165016501650165, 0.019801980198019802, 0.0242024202420242, 0.02145214521452145, 0.02035203520352035, 0.018151815181518153, 0.022002200220022004, 0.018151815181518153, 0.0231023102310231, 0.0176017601760176, 0.023652365236523653, 0.019251925192519254, 0.018151815181518153, 0.014301430143014302, 0.0231023102310231, 0.022552255225522552, 0.023652365236523653, 0.02035203520352035, 0.019251925192519254, 0.019251925192519254, 0.02145214521452145, 0.018151815181518153, 0.0176017601760176, 0.018151815181518153, 0.0187018701870187, 0.020902090209020903, 0.013751375137513752, 0.02145214521452145, 0.02145214521452145, 0.0242024202420242, 0.018151815181518153, 0.022002200220022004, 0.0176017601760176, 0.0242024202420242, 0.02145214521452145, 0.022552255225522552, 0.0165016501650165, 0.019251925192519254],
"cec28-2014": [0.020902090209020903, 0.019801980198019802, 0.015401540154015401, 0.017051705170517052, 0.02035203520352035, 0.02035203520352035, 0.022002200220022004, 0.022552255225522552, 0.024752475247524754, 0.022552255225522552, 0.0187018701870187, 0.018151815181518153, 0.02145214521452145, 0.0231023102310231, 0.01595159515951595, 0.022552255225522552, 0.014301430143014302, 0.022552255225522552, 0.02035203520352035, 0.0176017601760176, 0.02035203520352035, 0.02035203520352035, 0.0187018701870187, 0.0176017601760176, 0.019801980198019802, 0.019801980198019802, 0.02035203520352035, 0.02035203520352035, 0.020902090209020903, 0.018151815181518153, 0.019251925192519254, 0.019801980198019802, 0.02145214521452145, 0.02035203520352035, 0.019801980198019802, 0.022552255225522552, 0.02145214521452145, 0.019251925192519254, 0.02145214521452145, 0.02145214521452145, 0.01595159515951595, 0.0165016501650165, 0.020902090209020903, 0.0242024202420242, 0.019251925192519254, 0.022002200220022004, 0.019251925192519254, 0.0165016501650165, 0.02145214521452145, 0.02035203520352035],
"cec29-2014": [0.020206362854686157, 0.026225279449699053, 0.011607910576096303, 0.018056749785038694, 0.01633705932932072, 0.024505588993981083, 0.019776440240756664, 0.021926053310404127, 0.012897678417884782, 0.018486672398968184, 0.026225279449699053, 0.021496130696474634, 0.024935511607910577, 0.013327601031814273, 0.008598452278589854, 0.013757523645743766, 0.026655202063628546, 0.018056749785038694, 0.01633705932932072, 0.010318142734307825, 0.027944969905417026, 0.025795356835769563, 0.026225279449699053, 0.023215821152192607, 0.022785898538263114, 0.01117798796216681, 0.02536543422184007, 0.01117798796216681, 0.014617368873602751, 0.021496130696474634, 0.014617368873602751, 0.026225279449699053, 0.021496130696474634, 0.01633705932932072, 0.027944969905417026, 0.02880481513327601, 0.02536543422184007, 0.012897678417884782, 0.023215821152192607, 0.016766981943250214, 0.012897678417884782, 0.027515047291487533, 0.024505588993981083, 0.01934651762682717, 0.01934651762682717, 0.022785898538263114, 0.02536543422184007, 0.02063628546861565, 0.013757523645743766, 0.02063628546861565],
"cec30-2014": [0.019583333333333335, 0.015416666666666667, 0.014583333333333334, 0.020416666666666666, 0.014583333333333334, 0.016666666666666666, 0.020833333333333332, 0.009166666666666667, 0.017916666666666668, 0.020416666666666666, 0.019583333333333335, 0.02375, 0.014583333333333334, 0.020416666666666666, 0.02666666666666667, 0.009583333333333333, 0.02375, 0.0275, 0.019583333333333335, 0.019583333333333335, 0.01875, 0.024583333333333332, 0.02625, 0.017083333333333332, 0.020416666666666666, 0.020416666666666666, 0.025, 0.02, 0.020833333333333332, 0.028333333333333332, 0.020833333333333332, 0.025416666666666667, 0.02666666666666667, 0.01625, 0.02, 0.01625, 0.015416666666666667, 0.024166666666666666, 0.02, 0.019583333333333335, 0.020833333333333332, 0.022916666666666665, 0.02, 0.014583333333333334, 0.022083333333333333, 0.01625, 0.019583333333333335, 0.02375, 0.018333333333333333, 0.020833333333333332],



***********************************
100-Dim:
"cec1-2014": [0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.005555555555555556, 0.007407407407407408, 0.007407407407407408, 0.007407407407407408, 0.012962962962962963, 0.007407407407407408, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.009259259259259259, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.005555555555555556, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.005555555555555556, 0.011111111111111112, 0.005555555555555556, 0.007407407407407408, 0.007407407407407408, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.009259259259259259, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.005555555555555556, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.011111111111111112, 0.011111111111111112],
"cec2-2014": [0.009259259259259259, 0.005555555555555556, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.005555555555555556, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.005555555555555556, 0.007407407407407408, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.007407407407407408, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.005555555555555556, 0.005555555555555556, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.007407407407407408, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.005555555555555556, 0.011111111111111112, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112],
"cec3-2014": [0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.007407407407407408, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.007407407407407408, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.005555555555555556, 0.012962962962962963, 0.007407407407407408, 0.009259259259259259, 0.011111111111111112, 0.007407407407407408, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.005555555555555556, 0.007407407407407408, 0.007407407407407408, 0.009259259259259259, 0.011111111111111112, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.005555555555555556, 0.007407407407407408],
"cec4-2014": [0.010299625468164793, 0.00749063670411985, 0.00749063670411985, 0.011235955056179775, 0.013108614232209739, 0.010299625468164793, 0.00749063670411985, 0.0056179775280898875, 0.009363295880149813, 0.009363295880149813, 0.009363295880149813, 0.013108614232209739, 0.013108614232209739, 0.011235955056179775, 0.0056179775280898875, 0.0056179775280898875, 0.009363295880149813, 0.0056179775280898875, 0.011235955056179775, 0.013108614232209739, 0.010299625468164793, 0.013108614232209739, 0.013108614232209739, 0.010299625468164793, 0.010299625468164793, 0.00749063670411985, 0.010299625468164793, 0.0056179775280898875, 0.009363295880149813, 0.00749063670411985, 0.009363295880149813, 0.009363295880149813, 0.009363295880149813, 0.011235955056179775, 0.013108614232209739, 0.013108614232209739, 0.013108614232209739, 0.00749063670411985, 0.011235955056179775, 0.0056179775280898875, 0.00749063670411985, 0.010299625468164793, 0.010299625468164793, 0.00749063670411985, 0.011235955056179775, 0.013108614232209739, 0.010299625468164793, 0.013108614232209739, 0.00749063670411985, 0.009363295880149813, 0.00749063670411985, 0.00749063670411985, 0.013108614232209739, 0.011235955056179775, 0.010299625468164793, 0.011235955056179775, 0.013108614232209739, 0.00749063670411985, 0.011235955056179775, 0.013108614232209739, 0.0056179775280898875, 0.013108614232209739, 0.013108614232209739, 0.009363295880149813, 0.013108614232209739, 0.00749063670411985, 0.011235955056179775, 0.0056179775280898875, 0.009363295880149813, 0.0056179775280898875, 0.009363295880149813, 0.013108614232209739, 0.009363295880149813, 0.00749063670411985, 0.009363295880149813, 0.013108614232209739, 0.00749063670411985, 0.013108614232209739, 0.011235955056179775, 0.013108614232209739, 0.011235955056179775, 0.0056179775280898875, 0.0056179775280898875, 0.009363295880149813, 0.0056179775280898875, 0.009363295880149813, 0.013108614232209739, 0.010299625468164793, 0.013108614232209739, 0.009363295880149813, 0.009363295880149813, 0.013108614232209739, 0.013108614232209739, 0.009363295880149813, 0.00749063670411985, 0.013108614232209739, 0.009363295880149813, 0.013108614232209739, 0.013108614232209739, 0.010299625468164793],
"cec5-2014": [0.011111111111111112, 0.009259259259259259, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.012962962962962963, 0.005555555555555556, 0.005555555555555556, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.012962962962962963, 0.005555555555555556, 0.007407407407407408, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.011111111111111112, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.005555555555555556, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.005555555555555556, 0.007407407407407408, 0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.005555555555555556, 0.011111111111111112, 0.007407407407407408, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963],
"cec6-2014": [0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.005555555555555556, 0.005555555555555556, 0.011111111111111112, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.005555555555555556, 0.007407407407407408, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.007407407407407408, 0.012962962962962963, 0.005555555555555556, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.007407407407407408, 0.007407407407407408, 0.005555555555555556, 0.012962962962962963, 0.005555555555555556, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.009259259259259259, 0.005555555555555556, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112],
"cec7-2014": [0.007407407407407408, 0.007407407407407408, 0.007407407407407408, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.009259259259259259, 0.005555555555555556, 0.005555555555555556, 0.005555555555555556, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.007407407407407408, 0.007407407407407408, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.009259259259259259, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.009259259259259259, 0.009259259259259259, 0.005555555555555556, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.007407407407407408, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.012962962962962963, 0.005555555555555556, 0.005555555555555556, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259],
"cec8-2014": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
"cec9-2014": [0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.005555555555555556, 0.011111111111111112, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.007407407407407408, 0.011111111111111112, 0.007407407407407408, 0.007407407407407408, 0.007407407407407408, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.005555555555555556, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.009259259259259259, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112],
"cec10-2014": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
"cec11-2014": [0.005555555555555556, 0.005555555555555556, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.011111111111111112, 0.005555555555555556, 0.007407407407407408, 0.009259259259259259, 0.011111111111111112, 0.007407407407407408, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.007407407407407408, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.005555555555555556, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.007407407407407408, 0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408],
"cec12-2014": [0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.007407407407407408, 0.009259259259259259, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.007407407407407408, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.005555555555555556, 0.007407407407407408, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.007407407407407408, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.005555555555555556],
"cec13-2014": [0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.005555555555555556, 0.005555555555555556, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.005555555555555556, 0.007407407407407408, 0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.007407407407407408, 0.011111111111111112, 0.007407407407407408, 0.005555555555555556, 0.009259259259259259, 0.007407407407407408, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.005555555555555556, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.011111111111111112, 0.009259259259259259, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.007407407407407408, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.005555555555555556, 0.009259259259259259, 0.007407407407407408, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112],
"cec14-2014": [0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.009259259259259259, 0.009259259259259259, 0.005555555555555556, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.011111111111111112, 0.005555555555555556, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.012962962962962963, 0.007407407407407408, 0.009259259259259259, 0.007407407407407408, 0.005555555555555556, 0.011111111111111112, 0.007407407407407408, 0.005555555555555556, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.007407407407407408, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.007407407407407408, 0.009259259259259259, 0.011111111111111112, 0.007407407407407408, 0.007407407407407408, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.005555555555555556, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.007407407407407408, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408],
"cec15-2014": [0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.005555555555555556, 0.009259259259259259, 0.009259259259259259, 0.009259259259259259, 0.005555555555555556, 0.009259259259259259, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.007407407407407408, 0.005555555555555556, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.011111111111111112, 0.009259259259259259, 0.007407407407407408, 0.009259259259259259, 0.005555555555555556, 0.009259259259259259, 0.012962962962962963, 0.005555555555555556, 0.009259259259259259, 0.007407407407407408, 0.005555555555555556, 0.011111111111111112, 0.011111111111111112, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.007407407407407408, 0.007407407407407408, 0.012962962962962963, 0.007407407407407408, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.007407407407407408, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.011111111111111112, 0.005555555555555556, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963],
"cec16-2014": [0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.005555555555555556, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.007407407407407408, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.005555555555555556, 0.005555555555555556, 0.011111111111111112, 0.007407407407407408, 0.007407407407407408, 0.011111111111111112, 0.009259259259259259, 0.005555555555555556, 0.011111111111111112, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.005555555555555556, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.005555555555555556, 0.007407407407407408, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.007407407407407408, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.005555555555555556, 0.007407407407407408, 0.007407407407407408, 0.011111111111111112, 0.007407407407407408, 0.007407407407407408, 0.009259259259259259, 0.011111111111111112, 0.012962962962962963, 0.011111111111111112, 0.009259259259259259, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.005555555555555556, 0.011111111111111112, 0.012962962962962963, 0.009259259259259259, 0.012962962962962963, 0.009259259259259259, 0.007407407407407408, 0.012962962962962963, 0.012962962962962963, 0.009259259259259259, 0.009259259259259259, 0.012962962962962963, 0.007407407407407408, 0.011111111111111112, 0.011111111111111112, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963, 0.012962962962962963],
"cec17-2014": [0.009302325581395349, 0.0069767441860465115, 0.009302325581395349, 0.0069767441860465115, 0.0069767441860465115, 0.0069767441860465115, 0.009302325581395349, 0.013953488372093023, 0.009302325581395349, 0.013953488372093023, 0.013953488372093023, 0.009302325581395349, 0.009302325581395349, 0.013953488372093023, 0.0069767441860465115, 0.009302325581395349, 0.009302325581395349, 0.0069767441860465115, 0.009302325581395349, 0.0069767441860465115, 0.013953488372093023, 0.009302325581395349, 0.0069767441860465115, 0.013953488372093023, 0.0069767441860465115, 0.0069767441860465115, 0.009302325581395349, 0.0069767441860465115, 0.009302325581395349, 0.009302325581395349, 0.0069767441860465115, 0.013953488372093023, 0.009302325581395349, 0.009302325581395349, 0.009302325581395349, 0.0069767441860465115, 0.009302325581395349, 0.0069767441860465115, 0.009302325581395349, 0.0069767441860465115, 0.013953488372093023, 0.009302325581395349, 0.009302325581395349, 0.0069767441860465115, 0.013953488372093023, 0.009302325581395349, 0.0069767441860465115, 0.013953488372093023, 0.0069767441860465115, 0.009302325581395349, 0.0069767441860465115, 0.0069767441860465115, 0.009302325581395349, 0.009302325581395349, 0.013953488372093023, 0.013953488372093023, 0.009302325581395349, 0.013953488372093023, 0.009302325581395349, 0.009302325581395349, 0.0069767441860465115, 0.0069767441860465115, 0.009302325581395349, 0.009302325581395349, 0.009302325581395349, 0.013953488372093023, 0.013953488372093023, 0.009302325581395349, 0.013953488372093023, 0.0069767441860465115, 0.013953488372093023, 0.013953488372093023, 0.013953488372093023, 0.009302325581395349, 0.009302325581395349, 0.013953488372093023, 0.009302325581395349, 0.009302325581395349, 0.013953488372093023, 0.013953488372093023, 0.013953488372093023, 0.009302325581395349, 0.009302325581395349, 0.0069767441860465115, 0.0069767441860465115, 0.0069767441860465115, 0.013953488372093023, 0.013953488372093023, 0.013953488372093023, 0.013953488372093023, 0.0069767441860465115, 0.009302325581395349, 0.0069767441860465115, 0.013953488372093023, 0.013953488372093023, 0.009302325581395349, 0.013953488372093023, 0.0069767441860465115, 0.0069767441860465115, 0.009302325581395349],
"cec18-2014": [0.015584415584415584, 0.01038961038961039, 0.01038961038961039, 0.01038961038961039, 0.015584415584415584, 0.015584415584415584, 0.015584415584415584, 0.015584415584415584, 0.003896103896103896, 0.003896103896103896, 0.01038961038961039, 0.015584415584415584, 0.015584415584415584, 0.015584415584415584, 0.003896103896103896, 0.01038961038961039, 0.003896103896103896, 0.01038961038961039, 0.003896103896103896, 0.003896103896103896, 0.01038961038961039, 0.01038961038961039, 0.015584415584415584, 0.01038961038961039, 0.015584415584415584, 0.01038961038961039, 0.01038961038961039, 0.015584415584415584, 0.015584415584415584, 0.003896103896103896, 0.01038961038961039, 0.01038961038961039, 0.01038961038961039, 0.01038961038961039, 0.015584415584415584, 0.01038961038961039, 0.003896103896103896, 0.01038961038961039, 0.01038961038961039, 0.01038961038961039, 0.003896103896103896, 0.003896103896103896, 0.01038961038961039, 0.01038961038961039, 0.01038961038961039, 0.01038961038961039, 0.01038961038961039, 0.01038961038961039, 0.003896103896103896, 0.003896103896103896, 0.015584415584415584, 0.003896103896103896, 0.003896103896103896, 0.01038961038961039, 0.003896103896103896, 0.015584415584415584, 0.003896103896103896, 0.01038961038961039, 0.015584415584415584, 0.01038961038961039, 0.015584415584415584, 0.003896103896103896, 0.015584415584415584, 0.015584415584415584, 0.003896103896103896, 0.003896103896103896, 0.003896103896103896, 0.015584415584415584, 0.003896103896103896, 0.015584415584415584, 0.01038961038961039, 0.015584415584415584, 0.003896103896103896, 0.003896103896103896, 0.01038961038961039, 0.015584415584415584, 0.003896103896103896, 0.01038961038961039, 0.003896103896103896, 0.015584415584415584, 0.003896103896103896, 0.015584415584415584, 0.015584415584415584, 0.01038961038961039, 0.003896103896103896, 0.01038961038961039, 0.015584415584415584, 0.01038961038961039, 0.01038961038961039, 0.01038961038961039, 0.015584415584415584, 0.015584415584415584, 0.003896103896103896, 0.003896103896103896, 0.015584415584415584, 0.01038961038961039, 0.01038961038961039, 0.01038961038961039, 0.01038961038961039, 0.003896103896103896],
"cec19-2014": [0.015508021390374332, 0.0071301247771836, 0.0035650623885918, 0.015508021390374332, 0.0071301247771836, 0.0071301247771836, 0.015508021390374332, 0.0071301247771836, 0.0071301247771836, 0.0106951871657754, 0.015508021390374332, 0.0106951871657754, 0.0106951871657754, 0.015508021390374332, 0.0106951871657754, 0.015508021390374332, 0.0071301247771836, 0.015508021390374332, 0.015508021390374332, 0.0071301247771836, 0.0106951871657754, 0.015508021390374332, 0.0071301247771836, 0.0106951871657754, 0.0106951871657754, 0.0071301247771836, 0.0071301247771836, 0.0035650623885918, 0.0071301247771836, 0.0035650623885918, 0.0035650623885918, 0.0035650623885918, 0.0106951871657754, 0.0035650623885918, 0.0071301247771836, 0.0106951871657754, 0.0035650623885918, 0.0106951871657754, 0.015508021390374332, 0.0106951871657754, 0.0071301247771836, 0.015508021390374332, 0.015508021390374332, 0.0035650623885918, 0.0106951871657754, 0.015508021390374332, 0.0071301247771836, 0.0106951871657754, 0.015508021390374332, 0.015508021390374332, 0.0035650623885918, 0.015508021390374332, 0.0106951871657754, 0.0071301247771836, 0.015508021390374332, 0.0071301247771836, 0.0035650623885918, 0.0071301247771836, 0.0106951871657754, 0.0106951871657754, 0.0106951871657754, 0.0106951871657754, 0.015508021390374332, 0.015508021390374332, 0.0035650623885918, 0.0106951871657754, 0.0106951871657754, 0.015508021390374332, 0.0035650623885918, 0.0106951871657754, 0.015508021390374332, 0.0106951871657754, 0.015508021390374332, 0.015508021390374332, 0.0106951871657754, 0.0035650623885918, 0.0071301247771836, 0.0106951871657754, 0.0035650623885918, 0.0035650623885918, 0.015508021390374332, 0.015508021390374332, 0.0035650623885918, 0.015508021390374332, 0.0035650623885918, 0.0035650623885918, 0.0106951871657754, 0.0035650623885918, 0.015508021390374332, 0.0106951871657754, 0.0071301247771836, 0.015508021390374332, 0.015508021390374332, 0.0035650623885918, 0.0071301247771836, 0.0106951871657754, 0.0106951871657754, 0.015508021390374332, 0.0106951871657754, 0.0106951871657754],
"cec20-2014": [0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.014285714285714285, 0.0035714285714285713, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.014285714285714285, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.0035714285714285713, 0.014285714285714285, 0.010714285714285714, 0.010714285714285714, 0.0035714285714285713, 0.010714285714285714, 0.014285714285714285, 0.014285714285714285, 0.0035714285714285713, 0.0035714285714285713, 0.010714285714285714, 0.010714285714285714, 0.0035714285714285713, 0.0035714285714285713, 0.010714285714285714, 0.010714285714285714, 0.014285714285714285, 0.010714285714285714, 0.0035714285714285713, 0.010714285714285714, 0.0035714285714285713, 0.010714285714285714, 0.0035714285714285713, 0.010714285714285714, 0.0035714285714285713, 0.0035714285714285713, 0.010714285714285714, 0.014285714285714285, 0.010714285714285714, 0.010714285714285714, 0.014285714285714285, 0.0035714285714285713, 0.0035714285714285713, 0.010714285714285714, 0.010714285714285714, 0.014285714285714285, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.0035714285714285713, 0.0035714285714285713, 0.010714285714285714, 0.0035714285714285713, 0.014285714285714285, 0.010714285714285714, 0.010714285714285714, 0.014285714285714285, 0.014285714285714285, 0.014285714285714285, 0.014285714285714285, 0.010714285714285714, 0.010714285714285714, 0.014285714285714285, 0.010714285714285714, 0.010714285714285714, 0.014285714285714285, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.014285714285714285, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.0035714285714285713, 0.010714285714285714, 0.010714285714285714, 0.014285714285714285, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.014285714285714285, 0.014285714285714285, 0.010714285714285714, 0.010714285714285714, 0.010714285714285714, 0.0035714285714285713, 0.010714285714285714, 0.010714285714285714, 0.0035714285714285713],
"cec21-2014": [0.007075471698113208, 0.0047169811320754715, 0.013443396226415095, 0.0047169811320754715, 0.013443396226415095, 0.018867924528301886, 0.0047169811320754715, 0.007075471698113208, 0.0047169811320754715, 0.013443396226415095, 0.0047169811320754715, 0.007075471698113208, 0.007075471698113208, 0.018867924528301886, 0.0047169811320754715, 0.013443396226415095, 0.0047169811320754715, 0.0047169811320754715, 0.013443396226415095, 0.013443396226415095, 0.013443396226415095, 0.013443396226415095, 0.007075471698113208, 0.0047169811320754715, 0.007075471698113208, 0.013443396226415095, 0.0047169811320754715, 0.007075471698113208, 0.018867924528301886, 0.018867924528301886, 0.007075471698113208, 0.018867924528301886, 0.013443396226415095, 0.0047169811320754715, 0.018867924528301886, 0.007075471698113208, 0.0047169811320754715, 0.007075471698113208, 0.013443396226415095, 0.0047169811320754715, 0.0047169811320754715, 0.0047169811320754715, 0.0047169811320754715, 0.007075471698113208, 0.018867924528301886, 0.007075471698113208, 0.018867924528301886, 0.018867924528301886, 0.013443396226415095, 0.013443396226415095, 0.007075471698113208, 0.018867924528301886, 0.018867924528301886, 0.013443396226415095, 0.018867924528301886, 0.0047169811320754715, 0.018867924528301886, 0.013443396226415095, 0.018867924528301886, 0.013443396226415095, 0.018867924528301886, 0.0047169811320754715, 0.0047169811320754715, 0.0047169811320754715, 0.007075471698113208, 0.0047169811320754715, 0.007075471698113208, 0.0047169811320754715, 0.007075471698113208, 0.0047169811320754715, 0.007075471698113208, 0.007075471698113208, 0.007075471698113208, 0.007075471698113208, 0.007075471698113208, 0.007075471698113208, 0.0047169811320754715, 0.018867924528301886, 0.0047169811320754715, 0.0047169811320754715, 0.018867924528301886, 0.0047169811320754715, 0.013443396226415095, 0.007075471698113208, 0.007075471698113208, 0.007075471698113208, 0.018867924528301886, 0.013443396226415095, 0.007075471698113208, 0.018867924528301886, 0.018867924528301886, 0.007075471698113208, 0.0047169811320754715, 0.007075471698113208, 0.0047169811320754715, 0.013443396226415095, 0.0047169811320754715, 0.013443396226415095, 0.007075471698113208, 0.007075471698113208],
"cec22-2014": [0.013636363636363636, 0.013636363636363636, 0.00909090909090909, 0.013636363636363636, 0.004545454545454545, 0.00909090909090909, 0.013636363636363636, 0.00909090909090909, 0.00909090909090909, 0.00909090909090909, 0.013636363636363636, 0.013636363636363636, 0.00909090909090909, 0.013636363636363636, 0.013636363636363636, 0.004545454545454545, 0.00909090909090909, 0.013636363636363636, 0.00909090909090909, 0.004545454545454545, 0.013636363636363636, 0.00909090909090909, 0.00909090909090909, 0.013636363636363636, 0.013636363636363636, 0.00909090909090909, 0.013636363636363636, 0.004545454545454545, 0.013636363636363636, 0.013636363636363636, 0.004545454545454545, 0.004545454545454545, 0.004545454545454545, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.004545454545454545, 0.004545454545454545, 0.004545454545454545, 0.004545454545454545, 0.00909090909090909, 0.004545454545454545, 0.00909090909090909, 0.004545454545454545, 0.00909090909090909, 0.013636363636363636, 0.004545454545454545, 0.004545454545454545, 0.00909090909090909, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.00909090909090909, 0.004545454545454545, 0.013636363636363636, 0.013636363636363636, 0.004545454545454545, 0.004545454545454545, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.00909090909090909, 0.004545454545454545, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.004545454545454545, 0.013636363636363636, 0.004545454545454545, 0.004545454545454545, 0.013636363636363636, 0.004545454545454545, 0.013636363636363636, 0.00909090909090909, 0.004545454545454545, 0.004545454545454545, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.004545454545454545, 0.004545454545454545, 0.004545454545454545, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.013636363636363636, 0.004545454545454545, 0.004545454545454545, 0.013636363636363636, 0.00909090909090909, 0.00909090909090909, 0.013636363636363636],
"cec23-2014": [0.008554842652001222, 0.009318667888787045, 0.010235258172930034, 0.01008249312557287, 0.009318667888787045, 0.009624197983501375, 0.012679498930644668, 0.011762908646501681, 0.01237396883593034, 0.010540788267644362, 0.010235258172930034, 0.010235258172930034, 0.00717995722578674, 0.00977696303085854, 0.009929728078215704, 0.009013137794072716, 0.011762908646501681, 0.010235258172930034, 0.011151848457073022, 0.008707607699358386, 0.008707607699358386, 0.008707607699358386, 0.011457378551787351, 0.011457378551787351, 0.007485487320501069, 0.010235258172930034, 0.011762908646501681, 0.010540788267644362, 0.010540788267644362, 0.009013137794072716, 0.009471432936144211, 0.009318667888787045, 0.011762908646501681, 0.011457378551787351, 0.010235258172930034, 0.012679498930644668, 0.010235258172930034, 0.011762908646501681, 0.009013137794072716, 0.010235258172930034, 0.01008249312557287, 0.009929728078215704, 0.010846318362358692, 0.009318667888787045, 0.009318667888787045, 0.010540788267644362, 0.007791017415215398, 0.01206843874121601, 0.01237396883593034, 0.009318667888787045, 0.010540788267644362, 0.01206843874121601, 0.009318667888787045, 0.011762908646501681, 0.009624197983501375, 0.007791017415215398, 0.010846318362358692, 0.008249312557286892, 0.009013137794072716, 0.010999083409715857, 0.009013137794072716, 0.009013137794072716, 0.010846318362358692, 0.011457378551787351, 0.011151848457073022, 0.010235258172930034, 0.010235258172930034, 0.011762908646501681, 0.011762908646501681, 0.011151848457073022, 0.007791017415215398, 0.011762908646501681, 0.009318667888787045, 0.0068744271310724105, 0.010846318362358692, 0.007791017415215398, 0.007791017415215398, 0.010540788267644362, 0.010540788267644362, 0.010235258172930034, 0.008402077604644058, 0.009318667888787045, 0.009318667888787045, 0.007791017415215398, 0.009929728078215704, 0.01206843874121601, 0.007485487320501069, 0.008249312557286892, 0.010693553315001528, 0.007791017415215398, 0.008096547509929728, 0.011151848457073022, 0.009471432936144211, 0.008402077604644058, 0.009318667888787045, 0.009318667888787045, 0.011762908646501681, 0.011457378551787351, 0.008707607699358386, 0.010540788267644362],
"cec24-2014": [0.011094224924012158, 0.008662613981762919, 0.006838905775075988, 0.005623100303951368, 0.008054711246200608, 0.011702127659574468, 0.011094224924012158, 0.009270516717325228, 0.012917933130699088, 0.008054711246200608, 0.008054711246200608, 0.012917933130699088, 0.011702127659574468, 0.011094224924012158, 0.009270516717325228, 0.009270516717325228, 0.008054711246200608, 0.011702127659574468, 0.008054711246200608, 0.012310030395136779, 0.007446808510638298, 0.010486322188449849, 0.008054711246200608, 0.009878419452887538, 0.007446808510638298, 0.006838905775075988, 0.011702127659574468, 0.011094224924012158, 0.007446808510638298, 0.011702127659574468, 0.011094224924012158, 0.011094224924012158, 0.009878419452887538, 0.012310030395136779, 0.011702127659574468, 0.010486322188449849, 0.010486322188449849, 0.006838905775075988, 0.010486322188449849, 0.009270516717325228, 0.010486322188449849, 0.008054711246200608, 0.008662613981762919, 0.008054711246200608, 0.008054711246200608, 0.012917933130699088, 0.006231003039513678, 0.009878419452887538, 0.012310030395136779, 0.009878419452887538, 0.011094224924012158, 0.011702127659574468, 0.011094224924012158, 0.007446808510638298, 0.011702127659574468, 0.011094224924012158, 0.010486322188449849, 0.011094224924012158, 0.006231003039513678, 0.008054711246200608, 0.007446808510638298, 0.012917933130699088, 0.009878419452887538, 0.011702127659574468, 0.009878419452887538, 0.010486322188449849, 0.012917933130699088, 0.009270516717325228, 0.009270516717325228, 0.010486322188449849, 0.010486322188449849, 0.006231003039513678, 0.011094224924012158, 0.012310030395136779, 0.010486322188449849, 0.008662613981762919, 0.012310030395136779, 0.008662613981762919, 0.006838905775075988, 0.011702127659574468, 0.012310030395136779, 0.010486322188449849, 0.010486322188449849, 0.009878419452887538, 0.009270516717325228, 0.012917933130699088, 0.010486322188449849, 0.012917933130699088, 0.009270516717325228, 0.008662613981762919, 0.012917933130699088, 0.011702127659574468, 0.011702127659574468, 0.009270516717325228, 0.007446808510638298, 0.008054711246200608, 0.010486322188449849, 0.008662613981762919, 0.012917933130699088, 0.011094224924012158],
"cec25-2014": [0.008333333333333333, 0.012962962962962963, 0.010185185185185186, 0.010185185185185186, 0.011574074074074073, 0.009259259259259259, 0.007407407407407408, 0.00787037037037037, 0.008796296296296297, 0.011111111111111112, 0.009722222222222222, 0.011574074074074073, 0.0125, 0.009722222222222222, 0.010185185185185186, 0.010648148148148148, 0.008796296296296297, 0.010648148148148148, 0.009259259259259259, 0.00787037037037037, 0.009722222222222222, 0.009722222222222222, 0.011574074074074073, 0.012037037037037037, 0.008796296296296297, 0.009259259259259259, 0.011111111111111112, 0.010648148148148148, 0.010648148148148148, 0.0125, 0.011111111111111112, 0.011111111111111112, 0.011111111111111112, 0.00787037037037037, 0.008796296296296297, 0.006481481481481481, 0.009722222222222222, 0.008796296296296297, 0.008796296296296297, 0.010648148148148148, 0.010648148148148148, 0.008796296296296297, 0.008796296296296297, 0.011574074074074073, 0.00787037037037037, 0.011111111111111112, 0.010185185185185186, 0.011111111111111112, 0.006944444444444444, 0.011111111111111112, 0.011111111111111112, 0.009259259259259259, 0.0125, 0.011111111111111112, 0.009722222222222222, 0.008333333333333333, 0.009259259259259259, 0.011111111111111112, 0.008796296296296297, 0.008333333333333333, 0.008796296296296297, 0.009722222222222222, 0.0060185185185185185, 0.011111111111111112, 0.012962962962962963, 0.011574074074074073, 0.010648148148148148, 0.010648148148148148, 0.00787037037037037, 0.012037037037037037, 0.00787037037037037, 0.011574074074074073, 0.009722222222222222, 0.012037037037037037, 0.010185185185185186, 0.012037037037037037, 0.008333333333333333, 0.011111111111111112, 0.008333333333333333, 0.010185185185185186, 0.011574074074074073, 0.011111111111111112, 0.010648148148148148, 0.006481481481481481, 0.006481481481481481, 0.010648148148148148, 0.011111111111111112, 0.012037037037037037, 0.009722222222222222, 0.010185185185185186, 0.008796296296296297, 0.010185185185185186, 0.00787037037037037, 0.012037037037037037, 0.009259259259259259, 0.010648148148148148, 0.008333333333333333, 0.011574074074074073, 0.011111111111111112, 0.010648148148148148],
"cec26-2014": [0.009722222222222222, 0.01087962962962963, 0.007407407407407408, 0.00949074074074074, 0.01087962962962963, 0.006481481481481481, 0.010648148148148148, 0.011574074074074073, 0.00949074074074074, 0.009259259259259259, 0.00949074074074074, 0.01087962962962963, 0.011574074074074073, 0.009722222222222222, 0.008101851851851851, 0.009722222222222222, 0.01087962962962963, 0.010185185185185186, 0.009722222222222222, 0.009259259259259259, 0.010648148148148148, 0.008333333333333333, 0.011574074074074073, 0.006944444444444444, 0.008333333333333333, 0.009027777777777777, 0.010416666666666666, 0.012037037037037037, 0.009722222222222222, 0.010648148148148148, 0.011111111111111112, 0.008564814814814815, 0.009953703703703704, 0.008796296296296297, 0.01087962962962963, 0.009259259259259259, 0.008101851851851851, 0.008796296296296297, 0.010416666666666666, 0.00787037037037037, 0.008796296296296297, 0.00949074074074074, 0.010185185185185186, 0.011574074074074073, 0.011111111111111112, 0.01087962962962963, 0.010185185185185186, 0.009027777777777777, 0.011111111111111112, 0.01087962962962963, 0.008796296296296297, 0.011111111111111112, 0.010416666666666666, 0.008101851851851851, 0.011111111111111112, 0.011805555555555555, 0.010648148148148148, 0.009953703703703704, 0.01087962962962963, 0.008796296296296297, 0.008333333333333333, 0.008333333333333333, 0.01087962962962963, 0.011111111111111112, 0.008796296296296297, 0.009259259259259259, 0.010648148148148148, 0.00949074074074074, 0.011574074074074073, 0.008333333333333333, 0.012037037037037037, 0.012037037037037037, 0.00949074074074074, 0.010648148148148148, 0.00787037037037037, 0.00949074074074074, 0.011111111111111112, 0.011805555555555555, 0.011111111111111112, 0.011805555555555555, 0.008333333333333333, 0.009027777777777777, 0.010648148148148148, 0.01087962962962963, 0.01087962962962963, 0.010416666666666666, 0.00949074074074074, 0.011574074074074073, 0.012037037037037037, 0.011805555555555555, 0.011574074074074073, 0.010185185185185186, 0.008564814814814815, 0.010416666666666666, 0.009722222222222222, 0.009953703703703704, 0.009722222222222222, 0.008564814814814815, 0.01087962962962963, 0.00949074074074074],
"cec27-2014": [0.011522633744855968, 0.010905349794238683, 0.011522633744855968, 0.010082304526748971, 0.010699588477366255, 0.010082304526748971, 0.010493827160493827, 0.00823045267489712, 0.008436213991769548, 0.008024691358024692, 0.009465020576131687, 0.00905349794238683, 0.01213991769547325, 0.010082304526748971, 0.008436213991769548, 0.010699588477366255, 0.007818930041152264, 0.011111111111111112, 0.01131687242798354, 0.009670781893004115, 0.010082304526748971, 0.007407407407407408, 0.011728395061728396, 0.01131687242798354, 0.00905349794238683, 0.009670781893004115, 0.009876543209876543, 0.008436213991769548, 0.010905349794238683, 0.011522633744855968, 0.00905349794238683, 0.009465020576131687, 0.008024691358024692, 0.008847736625514403, 0.009465020576131687, 0.010699588477366255, 0.011728395061728396, 0.012757201646090535, 0.007818930041152264, 0.008436213991769548, 0.007407407407407408, 0.010493827160493827, 0.007613168724279836, 0.009465020576131687, 0.011522633744855968, 0.012551440329218106, 0.009670781893004115, 0.007613168724279836, 0.010082304526748971, 0.00905349794238683, 0.011522633744855968, 0.01131687242798354, 0.008024691358024692, 0.011934156378600824, 0.012757201646090535, 0.011111111111111112, 0.009465020576131687, 0.01131687242798354, 0.009465020576131687, 0.00823045267489712, 0.012757201646090535, 0.011111111111111112, 0.00823045267489712, 0.008641975308641974, 0.010082304526748971, 0.010493827160493827, 0.01131687242798354, 0.009465020576131687, 0.006584362139917695, 0.008641975308641974, 0.0102880658436214, 0.01213991769547325, 0.010493827160493827, 0.011728395061728396, 0.009465020576131687, 0.011111111111111112, 0.009259259259259259, 0.008436213991769548, 0.010082304526748971, 0.011111111111111112, 0.010493827160493827, 0.010905349794238683, 0.0102880658436214, 0.009259259259259259, 0.010905349794238683, 0.0102880658436214, 0.008024691358024692, 0.010905349794238683, 0.010905349794238683, 0.010905349794238683, 0.010493827160493827, 0.008436213991769548, 0.009465020576131687, 0.011522633744855968, 0.009465020576131687, 0.009465020576131687, 0.010493827160493827, 0.008847736625514403, 0.009876543209876543, 0.010905349794238683],
"cec28-2014": [0.009876543209876543, 0.008436213991769548, 0.008436213991769548, 0.009465020576131687, 0.012757201646090535, 0.010699588477366255, 0.009670781893004115, 0.011934156378600824, 0.010082304526748971, 0.011728395061728396, 0.008847736625514403, 0.010082304526748971, 0.011522633744855968, 0.009876543209876543, 0.006790123456790123, 0.0102880658436214, 0.009670781893004115, 0.01131687242798354, 0.011111111111111112, 0.009259259259259259, 0.009670781893004115, 0.011111111111111112, 0.009876543209876543, 0.008847736625514403, 0.011728395061728396, 0.010493827160493827, 0.0102880658436214, 0.00720164609053498, 0.009876543209876543, 0.008847736625514403, 0.010905349794238683, 0.00905349794238683, 0.010905349794238683, 0.008641975308641974, 0.010082304526748971, 0.009670781893004115, 0.011111111111111112, 0.01131687242798354, 0.010905349794238683, 0.011111111111111112, 0.010699588477366255, 0.009259259259259259, 0.009259259259259259, 0.011728395061728396, 0.008024691358024692, 0.010082304526748971, 0.008847736625514403, 0.010082304526748971, 0.011728395061728396, 0.010082304526748971, 0.011522633744855968, 0.011728395061728396, 0.009465020576131687, 0.011728395061728396, 0.009876543209876543, 0.0102880658436214, 0.010493827160493827, 0.010905349794238683, 0.00905349794238683, 0.010082304526748971, 0.009876543209876543, 0.00823045267489712, 0.011728395061728396, 0.0102880658436214, 0.008436213991769548, 0.010699588477366255, 0.009465020576131687, 0.010699588477366255, 0.008847736625514403, 0.01131687242798354, 0.009876543209876543, 0.008436213991769548, 0.00823045267489712, 0.00905349794238683, 0.009465020576131687, 0.00823045267489712, 0.009670781893004115, 0.010699588477366255, 0.007818930041152264, 0.010905349794238683, 0.00905349794238683, 0.010082304526748971, 0.009876543209876543, 0.00905349794238683, 0.0102880658436214, 0.009259259259259259, 0.011934156378600824, 0.008641975308641974, 0.008641975308641974, 0.012551440329218106, 0.010493827160493827, 0.00905349794238683, 0.009465020576131687, 0.010082304526748971, 0.007407407407407408, 0.010493827160493827, 0.009465020576131687, 0.011934156378600824, 0.011522633744855968, 0.0102880658436214],
"cec29-2014": [0.011150047785919083, 0.01083147499203568, 0.010035043007327174, 0.00796431984708506, 0.005734310289901242, 0.010194329404268876, 0.015291494106403312, 0.011150047785919083, 0.005734310289901242, 0.012105766167569289, 0.012742911755336095, 0.015610066900286716, 0.008601465434851864, 0.010194329404268876, 0.010194329404268876, 0.007008601465434852, 0.014335775724753107, 0.011150047785919083, 0.007486460656259956, 0.007645747053201656, 0.010194329404268876, 0.008601465434851864, 0.011150047785919083, 0.010353615801210577, 0.007008601465434852, 0.009238611022618668, 0.01083147499203568, 0.00828289264096846, 0.0133800573431029, 0.011468620579802484, 0.008920038228735267, 0.0136986301369863, 0.003822873526600828, 0.012742911755336095, 0.010194329404268876, 0.00796431984708506, 0.011150047785919083, 0.009557183816502071, 0.009238611022618668, 0.007008601465434852, 0.008601465434851864, 0.011627906976744186, 0.011150047785919083, 0.011150047785919083, 0.010194329404268876, 0.012424338961452692, 0.005734310289901242, 0.012105766167569289, 0.005097164702134438, 0.0136986301369863, 0.011468620579802484, 0.006212169480726346, 0.0133800573431029, 0.007645747053201656, 0.008601465434851864, 0.0135393437400446, 0.009238611022618668, 0.010194329404268876, 0.012742911755336095, 0.011787193373685887, 0.011787193373685887, 0.010194329404268876, 0.01083147499203568, 0.005734310289901242, 0.009238611022618668, 0.00669002867155145, 0.007645747053201656, 0.009557183816502071, 0.009557183816502071, 0.008920038228735267, 0.00669002867155145, 0.011787193373685887, 0.011150047785919083, 0.0063714558776680474, 0.014017202930869704, 0.011150047785919083, 0.007645747053201656, 0.013061484549219496, 0.007645747053201656, 0.014176489327811405, 0.00796431984708506, 0.014335775724753107, 0.012742911755336095, 0.007645747053201656, 0.006212169480726346, 0.00796431984708506, 0.014335775724753107, 0.008920038228735267, 0.011468620579802484, 0.014335775724753107, 0.0133800573431029, 0.009238611022618668, 0.012583625358394393, 0.0063714558776680474, 0.008601465434851864, 0.010194329404268876, 0.009716470213443773, 0.003822873526600828, 0.008601465434851864, 0.011150047785919083],
"cec30-2014": [0.008541600759253401, 0.012337867763366024, 0.011072445428661816, 0.011072445428661816, 0.008225245175577349, 0.013603290098070231, 0.008857956342929452, 0.012179689971527997, 0.008541600759253401, 0.010439734261309713, 0.007908889591901298, 0.007276178424549193, 0.013919645681746282, 0.009174311926605505, 0.012337867763366024, 0.010597912053147737, 0.010439734261309713, 0.009174311926605505, 0.01249604555520405, 0.01012337867763366, 0.009490667510281556, 0.009807023093957609, 0.013919645681746282, 0.005378044922492882, 0.013286934514394179, 0.010439734261309713, 0.010439734261309713, 0.009174311926605505, 0.008225245175577349, 0.010756089844985764, 0.009807023093957609, 0.009490667510281556, 0.011388801012337867, 0.009807023093957609, 0.011072445428661816, 0.010756089844985764, 0.01091426763682379, 0.013128756722556154, 0.006643467257197089, 0.008225245175577349, 0.007276178424549193, 0.012337867763366024, 0.007276178424549193, 0.01170515659601392, 0.008225245175577349, 0.007592534008225245, 0.009174311926605505, 0.008225245175577349, 0.014236001265422335, 0.0060107560898449855, 0.012021512179689971, 0.01012337867763366, 0.012654223347042075, 0.009174311926605505, 0.008857956342929452, 0.009965200885795633, 0.011546978804175894, 0.012021512179689971, 0.009174311926605505, 0.008541600759253401, 0.010439734261309713, 0.013286934514394179, 0.01012337867763366, 0.008225245175577349, 0.008541600759253401, 0.007908889591901298, 0.008541600759253401, 0.008225245175577349, 0.008541600759253401, 0.005378044922492882, 0.01249604555520405, 0.011072445428661816, 0.013286934514394179, 0.006959822840873141, 0.008541600759253401, 0.01170515659601392, 0.010439734261309713, 0.009807023093957609, 0.01249604555520405, 0.008225245175577349, 0.007908889591901298, 0.008857956342929452, 0.013286934514394179, 0.009490667510281556, 0.014552356849098386, 0.010756089844985764, 0.011072445428661816, 0.009807023093957609, 0.008225245175577349, 0.009965200885795633, 0.007908889591901298, 0.008225245175577349, 0.009490667510281556, 0.010756089844985764, 0.010439734261309713, 0.010756089844985764, 0.011072445428661816, 0.012970578930718128, 0.005378044922492882, 0.008225245175577349],


"""