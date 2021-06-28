from multiprocessing import Process, Lock
import time

lock = Lock()

def f(i):
    lock.acquire()
    try:
        print('hello world', i)
    finally:
        pass

def q(l):
    #l.release()
    print('bye world')


if __name__ == '__main__':

    for num in range(10):
        Process(target=f, args=(num, )).start()
        Process(target=q, args=(lock, )).start()
    for _ in range(10):
        time.sleep(1)
        lock.release()
