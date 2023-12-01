# -*- coding: utf-8 -*-
__author__ = 'Guy-Porat'

import socket
import os
from sys import argv
import tkFileDialog

Size_Header_Length = 30
Debug = True
default_port = 8820

def main():
    port = default_port

    if len(argv)==2:
        try:
            port = int(argv[1])
        except ValueError:
            print ("Please enter a numerical value between 0 and 65,536")
    if Debug: print "Bonding port " + str(port)

    try:
        ser_sock = socket.socket()
        ser_sock.bind(("0.0.0.0", port))
        ser_sock.listen(1)
        (receiver_sock, receiver_address) = ser_sock.accept()
        if Debug: print "connected to Receiver console. " + str(receiver_address)
        path2file = ask4file(True)
        send_data(return_file_name(path2file),receiver_sock)

        """Checks whether there is already a file with the same name at the receiver's working directory (to avoid over-writing it)"""
        print ("Awaiting confirmation...")
        if receiver_sock.recv(2) != "OK":
            print "It seems that a file of the same name already exists at the Receiver's working directory!"
            input()
            exit()
        send_file_by_parts(path2file, receiver_sock)

        """Checking with the receiver to see that the file sizes are the same at both sides"""
        if int(receiver_sock.recv(Size_Header_Length)) == os.stat(path2file).st_size:
            receiver_sock.send("TS")
            print "Receiver sent receive authentication, transfer successful!"
        else:
            receiver_sock.send("TF")
            print "It seems like something went wrong during the transfer!"
        receiver_sock.close()
        ser_sock.close()
    except socket.error:
        print "Receiver Disconnected Unexpectedly!"


"""if gui is true, it will ask for the file graphically
    else, it'll use the console"""
def ask4file(gui):
    if gui:
        file_path_string = tkFileDialog.askopenfilename()
    else:
        file_path_string = raw_input("Please enter the path to the file you wish to send: ")
        while (os.path.exists(path2file) != True):
            path2file = raw_input("Please enter the path to the file you wish to send: ")
    return file_path_string


def send_file_by_parts (path, socket):
    """should be used in collaboration with receive_file func! sends the file and returns True if it was received in full"""
    data_len = int(os.stat(path).st_size)  # os.path.getsize(path))
    socket.send(str(data_len).zfill(Size_Header_Length))
    with open(path, "rb") as file:
        data = "not none"
        sent_kb_cnt = 0
        while data:
            data = file.read(1024)
            socket.send(data)
            sent_kb_cnt += 1
            if sent_kb_cnt % 10 == 0:
                print str(sent_kb_cnt) + "KB sent. " + str((data_len/1024)-sent_kb_cnt) +" more KBs to go!"
    print "File Sent"


def send_data (data, socket):
    """should be used in collaboration with receive_data func! sends the data and returns True if it was received in full"""
    data_len = str(len(data)).zfill(Size_Header_Length)
    data2send = data_len + str(data)
    socket.send(data2send)

def return_file_name (path):
    return os.path.basename(path)

if __name__ == '__main__':
    main()