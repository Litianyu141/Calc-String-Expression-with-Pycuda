import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule
import time

# kernel func

kernel_code = """
#include<stdio.h>
#include<math.h>

  extern "C" {

   __global__ void add(float *result, float *para1, float *para2, const int N)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N) {
		result[i] = para1[i] + para2[i];
	}
}
__global__ void minus(float *result, float *para1, float *para2, const int N)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N) {
		result[i] = para1[i] - para2[i];
	}
}
__global__ void multiply(float *result, float *para1, float *para2, const int N)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N) {
		result[i] = para1[i] * para2[i];
	}
}
__global__ void divi(float *result, float *para1, float *para2, const int N)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N) {
		if (para2[i] == 0)
			result[i] = 0;//Do you have any good solution
		else
			result[i] = para1[i] / para2[i];

	}
}

__global__ void power(float *result, float *para1, int *para2, const int N)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N) {
		int count = para2[i];
		float result1 = 0;
		result1 = para1[i];
		//printf("count = %d", count);
		//printf("result1 = %f", result1);
		while (count > 1) {
			result1 = result1 * para1[i];
			count--;
		}
		result[i] = result1;
	}

}
__global__ void greater_op(float *result, float *para1, float *para2, const int N)
{

	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N) {
		if (para1[i] > para2[i])
            {
			result[i] = para1[i];
		}
	else
		result[i] = para2[i];
}


}
__global__ void lesser_op(float *result, float *para1, float *para2, const int N)
{

	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N) {
		if (para1[i] < para2[i])
	{
			result[i] = para1[i];
		}
	else
		result[i] = para2[i];
}


}

__global__ void if_op(float *result, float *para1, float *para2, const int N)
{

	return ;






}//There`s two arity
__global__ void and_op(float *result, float *para1, float *para2, const int N)
{

	return ;






}
__global__ void or_op(float *result, float *para1, float *para2, const int N)
{

	return ;






}
__global__ void sqrt_op(float *result, float *para1, const int N)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	float new_guess;
	float last_guess;
	float number = para1[i];

	if (number < 0) {
		result[i] = para1[i];
	}
	else if (number == 0) result[i] = 0;

	else {
		new_guess = 1;

		do {
			last_guess = new_guess;
			new_guess = (last_guess + number / last_guess) / 2;
		} while (new_guess != last_guess);
		result[i] = new_guess;
	}

}
 }//extern
"""
# init datasets


# init opeartors
mod = SourceModule(kernel_code, no_extern_c=True)
add = mod.get_function("add")
minus = mod.get_function("minus")
multiply = mod.get_function("multiply")
divi = mod.get_function("divi")
power = mod.get_function("power")
sqrt_op = mod.get_function("sqrt_op")
# init expression
expression = "a**i+b-c/a+b-c+(a+b)/c*a+b-c+a**i+b-c/a+b-c+(a+b)/c*a+b-c-a**i+b-c/a+b-c+(a+b)/c*a+b-c-a**i+b-c/a+b-c+(a+b)/c*a+b-c+a**i+b-c/a+b-c+(a+b)/c*a+b-c-a**i+b-c/a+b-c+(a+b)/c*a+b-c"
print("prototype:", expression)

# init data
N = 10000000  # This num is used for standing for dimmensionality and a placeholder for the amount of threads

a = np.random.randint(1, 10, size=(N, 1))
a = a.astype(np.float32)
b = np.random.randint(1, 10, size=(N, 1))
b = b.astype(np.float32)
c = np.random.randint(1, 10, size=(N, 1))
c = c.astype(np.float32)
index = 0.5

print("a:", a,end='')
print("b:", b,end='')
print("c:", c,end='')


# fun to deal with illegal expr,power expr and minus expr
def deal_minus_num(str1):
    i = 0
    str_list = list(str1)  # 字符串转list
    length = len(str_list)
    if str_list[i] == '-':
        if str_list[i + 1].isalpha():
            str_list.insert(i, '(')
            str_list.insert(i + 1, '0')
            str_list.insert(i + 4, ')')

    for i, element in enumerate(str_list):
        if str_list[i] == '-':
            if (str_list[i - 1] == '('):
                str_list.insert(i, '0')
            elif (str_list[i - 1] == "+") or (str_list[i - 1] == "-") or (str_list[i - 1] == "*") or (
                    str_list[i - 1] == "/"):
                str_list.insert(i, '0')
        elif str_list[i] == ' ':
            str_list.pop(i)
        elif str_list[i] == '*':
            if str_list[i + 1] == '*':
                str_list.pop(i)
                str_list[i] = '^'

    str_fin = ''.join(str_list)
    return str_fin


expression = deal_minus_num(expression)


# Medial expressions to suffix expressions
def middle2behind(expresssion):
    result1 = []
    stack1 = []
    for item in expression:
        if item.isalpha() or item.isdigit():
            result1.append(item)
        else:
            if len(stack1) == 0:
                stack1.append(item)
            elif item in '^':
                stack1.append(item)
            elif item in '*/(':
                stack1.append(item)
            elif item == ')':
                t = stack1.pop()
                while t != '(':
                    result1.append(t)
                    t = stack1.pop()

            elif item in '+-' and stack1[len(stack1) - 1] in '*/^':
                if stack1.count('(') == 0:
                    while stack1:
                        result1.append(stack1.pop())
                else:
                    t = stack1.pop()
                    while t != '(':
                        result1.append(t)
                        t = stack1.pop()
                    stack1.append('(')
                stack1.append(item)
            else:
                stack1.append(item)

    while stack1:
        result1.append(stack1.pop())

    return "".join(result1)


expr = middle2behind(expression)  # This is the expression

print("middle to suffix:", expr)

begin = time.time()  # Let`s calc the time of CPU


# the fuc to avoid 0 divisor

def divide(tensor1, tensor2):
    result = np.zeros_like(tensor2)
    for i in range(0, len(tensor2)):
        if tensor2[i] == 0:
            result[i] = 0
            print("unforunately")
        else:
            result[i] = tensor1[i] / tensor2[i]
    return result


# calc postfix expr
def calculate_postfix(postfix, tensors1, tensors2, tensors3):
    stack_cpu = []  # 用list模拟栈的后进先出
    for p in postfix:
        if p in '+-*/^':  # operator
            value_2 = stack_cpu.pop()  # first operand
            value_1 = stack_cpu.pop()  # second oprand
            if p == '+':
                result_cpu = value_1 + value_2
            elif p == '-':
                result_cpu = value_1 - value_2
            elif p == '*':
                result_cpu = value_1 * value_2
            elif p == '/':
                result_cpu = divide(value_1, value_2)
            elif p == '^':
                result_cpu = value_1
                for y in range(value_2 - 1):
                    result_cpu = result_cpu * value_1

            stack_cpu.append(result_cpu)
        elif p == 'a':
            stack_cpu.append(tensors1)

        elif p == 'b':
            stack_cpu.append(tensors2)

        elif p == 'c':
            stack_cpu.append(tensors3)

        elif p == 'i':
            num = int(index)
            stack_cpu.append(num)
        elif p.isdigit():
            num = int(p)
            stack_cpu.append(num)

    return stack_cpu.pop()


# excute calc with CPU
d = calculate_postfix(expr, a, b, c)

end = time.time() - begin
print("cpu time is:", end)

# init arrays for storing and prepare for CUDA program

result = np.zeros_like(a)
result = result.astype(np.float32)
marker = np.zeros_like(a)  # stands for d_res pointer
stack = []  # Calculating suffix expressions
stack_pointer = []  # store d_res device pointer


# start calc with CUDA

def isAString(obj):
    return isinstance(obj, str)


d_a = drv.mem_alloc(a.nbytes)
drv.memcpy_htod(d_a, a)

d_b = drv.mem_alloc(b.nbytes)
drv.memcpy_htod(d_b, b)

d_c = drv.mem_alloc(c.nbytes)
drv.memcpy_htod(d_c, c)

start = time.time()
for i, element in enumerate(expr):
    if element in '+-*/^':
        d_arg1 = stack.pop()

        d_arg2 = stack.pop()

        if element == '+':
            d_res = drv.mem_alloc(result.nbytes)
            add(d_res, d_arg2, d_arg1, np.int32(N), grid=((N - 1) // 128 + 1, 1), block=(128, 1, 1))
            d_resu = d_res
            stack.append(d_resu)

        elif element == '-':
            d_res = drv.mem_alloc(result.nbytes)
            minus(d_res, d_arg2, d_arg1, np.int32(N), grid=((N - 1) // 128 + 1, 1), block=(128, 1, 1))
            d_resu = d_res
            stack.append(d_resu)

        elif element == '*':
            d_res = drv.mem_alloc(result.nbytes)
            multiply(d_res, d_arg2, d_arg1, np.int32(N), grid=((N - 1) // 128 + 1, 1), block=(128, 1, 1))
            d_resu = d_res
            stack.append(d_resu)

        elif element == '/':
            d_res = drv.mem_alloc(result.nbytes)
            divi(d_res, d_arg2, d_arg1, np.int32(N), grid=((N - 1) // 128 + 1, 1), block=(128, 1, 1))
            d_resu = d_res
            stack.append(d_resu)

        elif element == '^':
            d_res = drv.mem_alloc(result.nbytes)
            power(d_res, d_arg2, d_arg1, np.int32(N), grid=((N - 1) // 128 + 1, 1), block=(128, 1, 1))
            d_resu = d_res
            stack.append(d_resu)

    elif element == 'a':
        stack.append(d_a)

    elif element == 'b':
        stack.append(d_b)

    elif element == 'c':
        stack.append(d_c)

    elif element == 'i':
        y = int(index)
        h_i_y = np.full((N, 1), y, dtype=np.int32)
        d_i_y = drv.mem_alloc(h_i_y.nbytes)
        drv.memcpy_htod(d_i_y, h_i_y)
        stack.append(d_i_y)
    elif element.isdigit():
        y = int(element)
        h_y = np.full((N, 1), y, dtype=np.int32)
        d_y = drv.mem_alloc(h_y.nbytes)
        drv.memcpy_htod(d_y, h_y)
        stack.append(d_y)

d_result = stack.pop()

drv.memcpy_dtoh(result, d_result)

final = time.time() - start

print("\nGPU time is:", final)

print("cpu answer is:", d)

print("gpu answer is:", result)

print("error:", abs(result) - abs(d))
