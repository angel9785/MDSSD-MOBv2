# # import torch
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # import time
# # s1 = torch.cuda.Stream()
# # s2 = torch.cuda.Stream()
# # # Initialise cuda tensors here. E.g.:
# # A = torch.rand(1000, 1000, device = device)
# # B = torch.rand(1000, 1000, device = device)
# # # # Wait for the above tensors to initialise.
# # # for i in range(0,5):
# # i=0
# # start = time.time()
# # torch.cuda.synchronize()
# # with torch.cuda.stream(s1):
# #     start1 = time.time()
# #     print(start1-start)
# #     # i=i+10
# #
# #     D = torch.mm(B, B)
# #     start3 = time.time()
# #     print(start3-start1 )
# #     # print(i)
# #     # c = torch.mm(A, A)
# #     # c=i+1
# # with torch.cuda.stream(s2):
# #     start2 = time.time()
# #     print(start2-start3)
# #     # i=i+10
# #     C = torch.mm(A, A)
# #     start4 = time.time()
# #     print(start4-start2)
# #     # print(i)
# #     # D = torch.mm(B, B)
# #     # print("stream"+"dor"+str(c))
# # # Wait for C and D to be computed.
# # torch.cuda.synchronize()
# # h=D+C
# # start5 = time.time()
# # print(start5-start)
#
# # print(a)
# # print(b)
# # Do stuff with C and D.
# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# import time
# import datetime
# torch.cuda.empty_cache()
# # torch.cuda.empty()# s1 = torch.cuda.Stream()
# # s2 = torch.cuda.Stream()
# # Initialise cuda tensors here. E.g.:
# A = torch.rand(1000, 1000, device = device)
# B = torch.rand(1000, 1000, device = device)
# W = torch.rand((1000, 1000), device=device)
# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)
# torch.multiprocessing.freeze_support()
# from torch.multiprocessing import Pool, Process, set_start_method
# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass
# def me(results):
#     torch.multiprocessing.freeze_support()
#     # Construct data_loader, optimizer, etc.
#     for arg in range(0, 499):
#     # gc.collect()
#     # torch.cuda.empty_cache()
#     # stream = torch.cuda.Stream(device)
#     # stream.wait_stream(main_stream)
#     # with torch.cuda.stream(stream):
#         x = torch.rand((1000,), device=device)
#         m.put=(W * x)
#         results.append(m)
#     # return results
# # torch.cuda.reset_max_memory_allocated(device)
# # torch.cuda.reset_peak_memory_stats
# # torch.cuda.empty_cache()
# # torch.cuda.ipc_collect()
# results = []
# processes = []
# if __name__ == '__main__':
#
#     start_time = datetime.datetime.now()
# # mp.set_start_method('spawn')
#     results = mp.Queue()
#     p = mp.Process(target=me,args=(results,))
#     p.start()
#     m1=results.get()
#     # results.append(m1)
#     processes.append(p)
#
#     p1 = mp.Process(target=me,args=(results,))
#     p1.start()
#     m2 = results.get()
#     # results.append(m2)
#     processes.append(p1)
#
#     for p in processes:
#         p.join()
# # for i in range(0,999):
# #     x = torch.rand((1000,), device=device)
# #     m=W * x
# #     results.append(m)
# # torch.cuda.synchronize(device)
# # main_stream = torch.cuda.current_stream(device)
# # stream = torch.cuda.Stream(device)
# # for arg in range(0,499):
# #     # gc.collect()
# #     # torch.cuda.empty_cache()
# #     # stream = torch.cuda.Stream(device)
# #     # stream.wait_stream(main_stream)
# #     with torch.cuda.stream(stream):
# #         x = torch.rand((1000,), device=device)
# #         m=W * x
# #         results.append(m)
# # # torch.cuda.synchronize(device)
# # stream1 = torch.cuda.Stream(device)
# #
# # for arg in range(0, 499):
# # # stream = torch.cuda.Stream(device)
# # # stream.wait_stream(main_stream)
# #     with torch.cuda.stream(stream1):
# #         x = torch.rand((1000,), device=device)
# #         m = W * x
# #         results.append(m)
# #     # results.app
# #     # main_stream.wait_stream(stream)
# # stre/////s.append(m)
# d=torch.stack(results).sum().item()
# torch.cuda.synchronize(device)
#
# duration = datetime.datetime.now() - start_time
# memory = torch.cuda.max_memory_allocated(device)
# print('result:', d)
# print('time:', duration)
# print('memory:', memory)
# mport torch.multiprocessing as mp
# from model import MyModel
#


# if __name__ == '__main__':
#     num_processes = 4
#     model = MyModel()
#     # NOTE: this is required for the ``fork`` method to work
#     model.share_memory()
#     processes = []
#     for rank in range(num_processes):
#         p = mp.Process(target=train, args=(model,))
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()
# # # Wait for the above tensors to initialise.
# # for i in range(0,5):
# # i=0
# # start = time.time()
# # torch.cuda.synchronize()
# # with torch.cuda.stream(s1):
# # start1 = time.time()
# # # print(start?1)
# # # i=i+10
# #
# # D = torch.mm(B, B)
# # start3 = time.time()
# # print(start3-start1 )
# # # print(i)
# # # c = torch.mm(A, A)
# # # c=i+1
# # # with torch.cuda.stream(s2):
# # start2 = time.time()
# # print(start2-start3)
# # # i=i+10
# # C = torch.mm(A, A)
# # start4 = time.time()
# # print(start4-start2)
# # # print(i)
# # # D = torch.mm(B, B)
# # # print("stream"+"dor"+str(c))
# # # Wait for C and D to be computed.
# # # torch.cuda.synchronize()
# # h=D+C
# # start5 = time.time()
# # print(start5-start1)
# #
# # # # print(a)
# # # # print(b)
# # # Do stuff with C and D.
import multiprocessing as mp
import time
m=100
def foo(q,e,f):
    q.put(10)

    # e.clear()
    # m = q.get()
    # print(f.put(10))
    # return m
    # print(m)
def foo1(m,d,h):
    # h.wai/t(5)
    d.put(m)
    # h.wait()


    # h.clear()
    # print(m)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    q = mp.Queue()
    d= mp.Queue()
    f = mp.Queue()
    e= mp.Event()
    h = mp.Event()
    p = mp.Process(target=foo, args=(q,e,f))
    p.start()

    # m = q.get()
    p1 = mp.Process(target=foo1, args=(m,d,h))
    p1.start()

    m = q.get()
    # m3=f.get()
    #

    #


    m1 = d.get()
    print(m)
    # print(m)
    # del m  ### REQUIRED! Once you call e.set(), this memory is no longer valid. It can be random.

    # h.set()

    print(m1)
    p.join()
    p1.join()
# import torch.multiprocessing as mp
# # from model import MyModel
# # m=0
# # m=Queue(10)
# def foo(q):
#     q.put(10)
#     # print(m)
# def foo1(q):
#     q.put(mp.Queue(10))
#
#
# if __name__ == '__main__':
#     num_processes = 4
#     q = mp.Queue()
#     # model = MyModel()
#     # NOTE: this is required for the ``fork`` method to work
#     # model.share_memory()
#     # processes = []
#     # for rank in range(num_processes):
#     p = mp.Process(target=foo, args=(q,))
#     p.start()
#     p1 = mp.Process(target=foo1, args=(q,))
#     p1.start()
#         # processes.append(p)
#     # for p in processes:
#     p.join()
#     p1.join()