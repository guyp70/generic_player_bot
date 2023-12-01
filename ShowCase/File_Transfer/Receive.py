# -*- coding: utf-8 -*-
__author__ = 'Guy-Porat'

import socket
import os
from sys import argv


Size_Header_Length = 30
Debug = True
default_port = 8820


def main():
    port = default_port

    if len(argv) == 3:
        try:
            port = int(argv[1])
            ip = argv[2]

        except ValueError:
            print ("Please enter a numerical value between 0 and 65,536")
    else:
        ip = raw_input("Please enter the sender's ip: ")
    if Debug:
        print "Attempting connection to " + str((ip, port))

    receiver_sock = socket.socket()
    receiver_sock.connect((ip, port))
    if Debug:
        print "connected to Sender console. " + str((ip, port))
    file_name = receive_data(receiver_sock)

    """Checks whether there is already a file with the same name (to avoid over-writing it)"""
    if os.path.exists(file_name):
        receiver_sock.send("NO")
        print "It seems that a file of the same name already exists at the working directory!"
        input()
        exit()
    receiver_sock.send("OK")
    receive_file_by_parts(file_name, receiver_sock)

    """Checking with the sender to see that the file sizes are the same at both sides"""
    rec_file_len = int(os.stat(file_name).st_size)  # os.path.getsize(path))
    receiver_sock.send(str(rec_file_len).zfill(Size_Header_Length))
    if receiver_sock.recv(len("TS")) == "TS":
        print "File Received Successfully"
    else:
        print "It seems like something went wrong during the transfer!"

    receiver_sock.close()


def receive_data(socket):
    """should be used in collaboration with send_data func! receives data and verifies it was received in full"""
    data_len = int(socket.recv(Size_Header_Length))
    received_data = ""
    while data_len > len(received_data):
        received_data += str(socket.recv(data_len - len(received_data)))  # data_len - len(received_data) equals to what remains to be received
    return received_data


def receive_file_by_parts (path, socket):
    total_data_len = int(socket.recv(Size_Header_Length))
    data_received_len = 0
    with open(path, "wb") as file2write:
        while total_data_len > data_received_len:
            data = socket.recv(total_data_len-data_received_len)
            file2write.write(data)
            data_received_len += len(data)
            print str(data_received_len/1024) + "KB received. " + str((total_data_len-data_received_len)/1024) + " more KBs to go!"
    print "File Received"


if __name__ == '__main__':
    main()