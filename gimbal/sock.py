import socket


class socket():
    def __init__(self) -> None:
        # Host and port for the server
        self.host = '127.0.0.1'
        self.port = 3030
        self.data = None
    def tcp_server(self):
        # Create a socket object using IPv4 and TCP protocol
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        

        # Bind the socket to the address
        server_socket.bind((self.host, self.port))

        # Listen for incoming connections (the number specifies the max number of queued connections)
        server_socket.listen(5)
        print("Server is listening on {}:{}".format(self.host, self.port))
        try:
            while True:
                # Accept a connection
                client_socket, addr = server_socket.accept()
                print("Got a connection from {}".format(addr))

                # Receive data from the client
                self.data = client_socket.recv(1024)
                print("Received '{}' from client".format(self.data))

                # Close the connection
                client_socket.close()
        except KeyboardInterrupt:
            # Close the server socket
            server_socket.close()
            print("Server is shutting down...")
        

# if __name__ == '__main__':
#     tcp_server()
