import numpy as np
from nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h) #bir
        self.bzx = np.random.randn(h) #biz
        self.bnx = np.random.randn(h) #bin

        self.brh = np.random.randn(h) #bhr
        self.bzh = np.random.randn(h) #bhz
        self.bnh = np.random.randn(h) #bhn

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h)) #dbir
        self.dbzx = np.zeros((h)) # dbiz
        self.dbnx = np.zeros((h)) #dbin

        self.dbrh = np.zeros((h)) #dbhr
        self.dbzh = np.zeros((h)) #dbzh
        self.dbnh = np.zeros((h)) #dbhn

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()
        self.n_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
         """GRU cell forward.

    #     Input
    #     -----
    #     x: (input_dim)
    #         observation at current time-step.

    #     h_prev_t: (hidden_dim)
    #         hidden-state at previous time-step.

    #     Returns
    #     -------
    #     h_t: (hidden_dim)
    #         hidden state at current time-step.

    #     """
         self.x = x
         self.hidden = h_prev_t
        
         # Add your code here.
         # Define your variables based on the writeup using the corresponding
         # names below.

         # r = (np.dot(self.Wrx,self.x)) + (self.brx) + (np.dot(self.Wrh,self.hidden)) + (self.brh)
         # self.r = self.r_act.forward(r)
         # z = (np.dot(self.Wzx,self.x)) + self.bzx + (np.dot(self.Wzh,self.hidden)) + (self.bzh)
         # self.z = self.z_act.forward(z)
         # n = (np.dot(self.Wnx, self.x)) + self.bnx + (self.r * ((np.dot(self.Wnh,self.hidden)) + self.bnh))
         # self.n = self.h_act.forward(n)
         # h_t = ((1 - self.z) * self.n) + (self.z * self.hidden)

         self.g1 = np.dot(self.Wrx, x)
         self.g2 = self.g1 + self.brx
         self.g3 = np.dot(self.Wrh, h_prev_t)
         self.g4 = self.g3 + self.brh
         self.g5 = self.g2 + self.g4
         self.r = self.r_act.forward(self.g5)
         self.g7 = np.dot(self.Wzx, x)
         self.g8 = self.g7 + self.bzx
         self.g9 = np.dot(self.Wzh, h_prev_t)
         self.g10 = self.g9 + self.bzh
         self.g11 = self.g10 + self.g8
         self.z = self.z_act.forward(self.g11)
         print("Shape of z is", self.z.shape)
         self.g13 = np.dot(self.Wnx, x)
         self.g14 = self.g13 + self.bnx
         self.g15 = np.dot(self.Wnh, h_prev_t)
         self.g16 = self.g15 + self.bnh
         self.g17 = self.r * self.g16
         self.g18 = self.g14 + self.g17
         self.n = self.h_act.forward(self.g18)
         self.g20 = 1 - self.z
         self.g21 = self.g20 * self.n
         self.g22 = self.z * h_prev_t
         h_t = self.g21 + self.g22
         print("Shape of h_t is", h_t.shape)
        #  self.r = self.r_act.forward(np.dot(self.Wrx,self.x)+ self.brx + np.dot(self.Wrh,self.hidden) + self.brh)
        #  self.z = self.z_act.forward(np.dot(self.Wzx,self.x)+ self.bzx + np.dot(self.Wzh,self.hidden) + self.bzh)
        #  self.n = self.h_act.forward(np.dot(self.Wnx,self.x)+ self.bnx + self.r*(np.dot(self.Wnh,self.hidden) + self.bnh))
        #  h_t = (1-self.z)*self.n + self.z*self.hidden
        
         assert self.x.shape == (self.d,)
         assert self.hidden.shape == (self.h,)

         assert self.r.shape == (self.h,)
         assert self.z.shape == (self.h,)
         assert self.n.shape == (self.h,)
         assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.


         return h_t
    #     raise NotImplementedError

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                  the same time-step and derivative wrt loss from same layer at
                  next time-step.

        Returns
          -------
          dx: (input_dim)
              derivative of the loss wrt the input x.

          dh_prev_t: (hidden_dim)
              derivative of the loss wrt the input hidden h.

          """

        # SOME TIPS:
        # 1) Make sure the shapes of the calculated dWs and dbs match the initalized shapes of the respective Ws and bs
        # 2) When in doubt about shapes, please refer to the table in the writeup.
        # 3) Know that the autograder grades the gradients in a certain order, and the local autograder will tell you which gradient you are currently failing.
    
        # print("Shape of self.n is", self.n.shape)
        # print("Shape of self.hidden is", self.hidden.shape)
        # print("Shape of delta is", delta.shape)
        
        # dlzt = delta * (self.hidden - self.n)
        # dldn = delta *(1-self.z)

        # print("Shape of dLdz is", dlzt.shape)
        # print("Shape of dldn is", dldn.shape)
    
        # dnt = dldn * self.h_act.backward(self.n)
        # print("Shape of dnt is", dnt.shape)
        # dz = dlzt * self.z_act.backward(self.z)

        # print("Shape of x is", self.x.shape)
        # print("Shape of dnt is", dnt.shape)
        # print("Shape of r is", self.r.shape)
        # print("Shape of hidden is", self.hidden.shape)
        # self.dWnx = np.dot(dnt.reshape(-1,1),(self.x.reshape(-1,1).T))
        # print("Shape of dWnx is", self.dWnx.shape)
        # self.dbnx = np.sum(dnt,axis = 0)
        # self.dWnh = dnt.T * np.dot(self.r,self.hidden.T) # r * self.hidden
        # self.dbnh = dnt * self.r
        # dlrt = dnt.T * (np.dot(self.Wnh,self.hidden) + self.bnh)
        # print("Shape of dlrt is", dlrt.shape)
        # print("Shape of Wnh is", self.Wnh.shape)
        # print("Shape of bnh is", self.bnh.shape)
        # print("Shape of term1 is", np.dot(self.Wnh,self.hidden).shape)
        # print("Shape of term2 is", (np.dot(self.Wnh,self.hidden) + self.bnh).shape)
        # print("Shape of dz is", dz.shape)
        # self.dWzx = np.dot((dz.reshape(-1,1)),(self.x.reshape(-1,1).T))
        # self.dbzh = np.sum(dz,axis = 0)
        # self.dWzh = np.dot(dz.T, self.hidden.T)
        # self.dbzx = np.sum(dz,axis = 0)
    
        # dr = dlrt * self.r_act.backward(dlrt)
        # print("Shape of dr is", dr.shape)
        # self.dWrx = np.dot(dr.reshape(-1,1),(self.x.reshape(-1,1).T))    
        # self.dbrx = np.sum(dr.reshape(-1,1), axis = 1)
        # print("Shape of dlbrx is", self.dbrx.shape)
        # self.dWrh = np.dot(dr,self.hidden.T)
        # self.dbrh = np.sum(dr.reshape(-1,1), axis = 1)

        # dx = np.dot(self.Wnx.T, dnt.T) + np.dot(self.Wzx.T, dz.T) + np.dot(self.Wrx.T,dr)
        # dx = dx.T

        # dh_prev_t = np.dot(self.Wnh.T,(self.r * dnt).T) + np.dot(self.Wzh.T,dz.T) + np.dot(self.Wrh.T,dr) + (delta * self.z).T
        # print("Shape of dh_prev_t", dh_prev_t.shape)
        # print("Shape of dh_prev_t term 1", np.dot(self.Wnh.T,(self.r * dnt).T).shape)
        # print("Shape of dh_prev_t term 2", np.dot(self.Wzh.T,dz.T).shape)
        # print("Shape of dh_prev_t term 3", np.dot(self.Wzh.T,dz.T).shape)
        # print("Shape of dh_prev_t term 4", np.dot(self.Wrh.T,dr).shape)
        # print("Shape of dh_prev_t term 5", (delta * self.z).T.shape)
        # # dh_prev_t = dh_prev_t.reshape(1,-1) 
        # assert dx.shape == (self.d,)
        # assert dh_prev_t.shape == (self.h,)

        # return dx, dh_prev_t
        dz = (self.hidden-self.n)*delta 
        dn = (1-self.z)*delta

        dn_act = self.h_act.backward(dn)
        print("The shapw of dn_act is",dn_act.shape)

        temp = self.Wnh@self.hidden + self.bnh
        # self.dWnx = np.outer(dn_act, self.x)
        self.dWnx = np.dot(dn_act.reshape(-1, 1), self.x.reshape(1, -1))
        self.dbnx = dn_act

        dr = dn_act*self.g16
        
        self.dWnh = np.dot((dn_act * self.r).reshape(-1, 1), self.hidden.reshape(1, -1))
        self.dbnh = dn_act * self.r

        dz_act = self.z_act.backward(dz)
        self.dWzx = np.dot(dz_act.reshape(-1, 1), self.x.reshape(1, -1))
        self.dbzx = dz_act
        self.dWzh = np.dot(dz_act.reshape(-1, 1), self.hidden.reshape(1, -1))
        self.dbzh = dz_act

        dr_act = self.r_act.backward(dr)
        self.dWrx = np.dot(dr_act.reshape(-1, 1), self.x.reshape(1, -1))
        self.dbrx = dr_act
        self.dWrh = np.dot(dr_act.reshape(-1, 1), self.hidden.reshape(1, -1))
        self.dbrh = dr_act  

        dx = self.Wnx.T@dn_act + self.Wzx.T@dz_act + self.Wrx.T@dr_act  
        dh_prev_t = self.Wnh.T@(dn_act*self.r) + self.Wzh.T@dz_act + self.Wrh.T@dr_act + (delta*self.z)

        return dx, dh_prev_t

#     #     # Derivatives of the final output with respect to various intermediate variables
#     #     dh_t = delta
#     #     dn = dh_t * (1 - self.z)
#     #     dz = dh_t * (self.hidden - self.n)
#     #     dh_prev_t = dh_t * self.z
        
#     #     # Derivatives of the intermediate variables with respect to the activation functions
#     #     d_g18 = dn * self.h_act.backward(self.g18)
#     #     d_g11 = dz * self.z_act.backward(self.g11)
#     #     d_g5 = self.r * d_g18 * self.Wnh.T
#     #     print("Shape of d_g5", d_g5.shape)
#     #     print("Shape of x is.", self.x)
#     #     d_g5 += self.r_act.backward(self.g5) * np.dot(self.Wrh.T, d_g5)
        
#     #     # Derivatives with respect to the weights and biases
#     #     self.dWnx += np.outer(d_g18, self.x)
#     #     self.dWnh += np.outer(d_g18, self.r)
#     #     self.dbnx += d_g18
#     #     self.dbnh += d_g18 * self.r
        
#     #     self.dWzx += np.outer(d_g11, self.x)
#     #     self.dWzh += np.outer(d_g11, self.hidden)
#     #     self.dbzx += d_g11
#     #     self.dbzh += d_g11
        
#     #     # self.dWrx += np.outer(d_g5, self.x)
#     #     self.dWrx += np.dot(d_g5.reshape(-1, 1), self.x.reshape(1, -1))
#     #     self.dWrh += np.outer(d_g5, self.hidden)
#     #     self.dbrx += d_g5
#     #     self.dbrh += d_g5
        
#     #     # Derivatives with respect to the inputs
#     #     dx = np.dot(self.Wrx.T, d_g5) + np.dot(self.Wzx.T, d_g11) + np.dot(self.Wnx.T, d_g18)
#     #     dh_prev_t += np.dot(self.Wrh.T, d_g5) + np.dot(self.Wzh.T, d_g11) + d_g18 * self.r * self.Wnh.T
        
#     #     assert dx.shape == (self.d,)
#     #     assert dh_prev_t.shape == (self.h,)

#     #     
#     #     # return dx, dh_prev_t
#     #     raise NotImplementedError
    # def forward(self, x, h_prev_t):
    #     """GRU cell forward.

    #     Input
    #     -----
    #     x: (input_dim)
    #         observation at current time-step.

    #     h_prev_t: (hidden_dim)
    #         hidden-state at previous time-step.

    #     Returns
    #     -------
    #     h_t: (hidden_dim)
    #         hidden state at current time-step.

    #     """
    #     self.x = x
    #     self.hidden = h_prev_t
        
    #     # Add your code here.
    #     # Define your variables based on the writeup using the corresponding
    #     # names below.
    #     self.r = self.r_act.forward(np.dot(self.Wrx,self.x)+ self.brx + np.dot(self.Wrh,self.hidden) + self.brh)
    #     self.z = self.z_act.forward(np.dot(self.Wzx,self.x)+ self.bzx + np.dot(self.Wzh,self.hidden) + self.bzh)
    #     self.n = self.h_act.forward(np.dot(self.Wnx,self.x)+ self.bnx + self.r*(np.dot(self.Wnh,self.hidden) + self.bnh))
    #     h_t = (1-self.z)*self.n + self.z*self.hidden
        
    #     assert self.x.shape == (self.d,)
    #     assert self.hidden.shape == (self.h,)

    #     assert self.r.shape == (self.h,)
    #     assert self.z.shape == (self.h,)
    #     assert self.n.shape == (self.h,)
    #     assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

    #     return h_t
    #     raise NotImplementedError

    # def backward(self, delta):
    #     """GRU cell backward.

    #     This must calculate the gradients wrt the parameters and return the
    #     derivative wrt the inputs, xt and ht, to the cell.

    #     Input
    #     -----
    #     delta: (hidden_dim)
    #             summation of derivative wrt loss from next layer at
    #             the same time-step and derivative wrt loss from same layer at
    #             next time-step.

    #     Returns
    #     -------
    #     dx: (1, input_dim)
    #         derivative of the loss wrt the input x.

    #     dh_prev_t: (1, hidden_dim)
    #         derivative of the loss wrt the input hidden h.

    #     """
    #     # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
    #     #    when computing self.dWs...

 
    #     dLdz = delta*(self.hidden-self.n)

    #     dLdn = delta*(1-self.z)
        
    #     dn = dLdn*self.h_act.backward(self.n)
    #     dz = dLdz * self.z_act.backward(dLdz)

    #     self.x = self.x.reshape(self.d,1)
    #     self.hidden = self.hidden.reshape(self.h,1)
    #     print("Shape of self.hidden is", self.hidden.shape)


    #     print("Shape of dn.T:", dn.T.shape)
    #     print("Shape of self.x.T:", self.x.T.shape)

    #     dn = dn.reshape(-1, 1)  # Reshape dn to be a column vector
    #     print("Shape of dn is", dn.shape)
    #     self.dWnx += np.dot(dn, self.x.T)  # Using += for accumulating gradients if needed

        
    #     self.dbnx = np.sum(dn,axis=0)

    #     self.dWnh = dn.T*np.dot(self.r.reshape(-1,1),self.hidden.T)
    #     self.dbnh = dn*self.r

        
    #     dz = dz.reshape(-1, 1)
    #     self.dWzx = np.dot(dz,self.x.T)
    #     self.dbzx = np.sum(dz,axis=0)
    #     self.dWzh = np.dot(dz,self.hidden.T)
    #     self.dbzh = np.sum(dz,axis=0)
        
    #     dLdr = dn.T* (np.dot(self.Wnh,self.hidden)+self.bnh.reshape(-1,1))
        
    #     # dr = dLdr*np.expand_dims(self.r_act.backward(),axis =1)
    #     # Calculate the gradient of n with respect to r
    #     # Calculate the gradient of n with respect to r
    #     print("Shape of wnh is", self.Wnh.shape)
    #     dndr = np.dot(self.Wnh, self.hidden).reshape(-1) + self.bnh  # Shape: (hidden_dim,)
    #     print("Shape of dndr is:", dndr.shape)

    #     # Calculate dr using the chain rule: dL/dn * dn/dr * dact/dr
    #     # dndr * dn: element-wise multiplication, resulting in a vector of shape (hidden_dim,)
    #     # self.r_act.backward(self.r_act.A): derivative of the activation function w.r.t. its input, resulting in a vector of shape (hidden_dim,)
    #     # dr = (dndr * dn.flatten()) * self.r_act.backward(self.r_act.A).flatten()  # Assuming self.r_act.A stores the activation input, shape: (hidden_dim,)
    #     partial_dL_dr = dndr * dn.flatten()
    #     # Correct call to self.r_act.backward
    #     dr = partial_dL_dr * self.r_act.backward(partial_dL_dr)
    #     # Use partial_dL_dr as dLdA
    #     dr = dr.reshape(-1, 1)
    #     print("Shape of dr", dr.shape)
    #     # Reshape dr to ensure it is a column vector
    #     dr = dr.reshape(-1, 1)  # Shape: (hidden_dim, 1)
    #     print("Post Shape of dr", dr.shape)

    #     self.dWrx = np.dot(dr,self.x.T)
    #     self.dbrx = np.sum(dr,axis=1)
    #     self.dWrh = np.dot(dr,self.hidden.T)
    #     self.dbrh = np.sum(dr,axis=1)


    #     dx = np.dot(self.Wnx.T,dn) + np.dot(self.Wzx.T,dz) + np.dot(self.Wrx.T,dr)
    #     dx = dx.T
        
    #     print("Shape of term1:", np.dot(self.Wnh.T, (self.r.reshape(1, -1) * dn).T).shape)
    #     print("Shape of term2:", np.dot(self.Wzh.T, dz).shape)
    #     print("Shape of term3:", np.dot(self.Wrh.T, dr).shape)
    #     print("Shape of term4:", (delta * self.z).T.shape)

    #     print("Shape of self.r is", self.r.shape)
    #     self.r = self.r.reshape(-1, 1)

    #     # Now perform element-wise multiplication and ensure the result is a column vector
    #     reset_dn = (self.r * dn)  # This should result in a shape of (3, 1)
    #     print("Shape of reset_dn is", reset_dn.shape)
    #     print("Shape of term1:", np.dot(self.Wnh.T, reset_dn).shape)

    #     delta_z = (delta * self.z).reshape(-1, 1)
    #     print("Shape of delta_z is", delta_z.shape)
    #     dh_prev_t = np.dot(self.Wnh.T, reset_dn) + np.dot(self.Wzh.T, dz) + np.dot(self.Wrh.T, dr) + delta_z
    #     print("Shape of dh_prev_t before reshaping:", dh_prev_t.shape)
    #     dh_prev_t = dh_prev_t.reshape(1,-1)
    #     print("Shape of dh_prev_t after reshaping:", dh_prev_t.shape)
    #     # 2) Transpose all calculated dWs...
        
    #     # 3) Compute all of the derivatives
    #     # 4) Know that the autograder grades the gradients in a certain order, and the
    #     #    local autograder will tell you which gradient you are currently failing.

    #     # ADDITIONAL TIP:
    #     # Make sure the shapes of the calculated dWs and dbs  match the
    #     # initalized shapes accordingly

        
    #     assert dx.shape == (1, self.d)

    #     # print("Shape of self.h is", leself.h))
       
    #     assert dh_prev_t.shape == (1, self.h)

        
    #     return dx, dh_prev_t

#     def forward(self, x, h):
#         # input:
#         #   - x: shape(input dim),  observation at current time-step
#         #   - h: shape(hidden dim), hidden-state at previous time-step
#         #
#         # output:
#         #   - h_t: hidden state at current time-step

#         self.x = x
#         self.hidden = h

#         # Add your code here.
#         # Define your variables based on the writeup using the corresponding
#         # names below.
#         self.z1 = np.dot(self.Wzh, h)
#         self.z2 = np.dot(self.Wzx, x)
#         self.z3 = self.z1 + self.z2
#         self.z4 = self.z_act.forward(self.z3)
#         self.z = self.z4

#         self.z5 = np.dot(self.Wrh, h)
#         self.z6 = np.dot(self.Wrx, x)
#         self.z7 = self.z5 + self.z6
#         self.z8 = self.r_act.forward(self.z7)
#         self.r = self.z8

#         self.z9 = self.z8 * h
#         self.z10 = np.dot(self.Wnh, self.z9)
#         self.z11 = np.dot(self.Wnx, x)
#         self.z12 = self.z10 + self.z11
#         self.z13 = self.h_act.forward(self.z12)
#         self.h_tilda = self.z13

#         self.z14 = 1 - self.z4
#         self.z15 = self.z14 * h
#         self.z16 = self.z4 * self.z13
#         self.z17 = self.z15 + self.z16
#         h_t = self.z17

#         assert self.x.shape == (self.d, )
#         assert self.hidden.shape == (self.h, )

#         assert self.r.shape == (self.h, )
#         assert self.z.shape == (self.h, )
#         assert self.h_tilda.shape == (self.h, )
#         assert h_t.shape == (self.h, )

#         return h_t


#     # This must calculate the gradients wrt the parameters and return the
#     # derivative wrt the inputs, xt and ht, to the cell.
#     def backward(self, delta):
#         # input:
#         #  - delta:  shape (hidden dim), summation of derivative wrt loss from next layer at
#         #            the same time-step and derivative wrt loss from same layer at
#         #            next time-step
#         # output:
#         #  - dx: Derivative of loss wrt the input x
#         #  - dh: Derivative  of loss wrt the input hidden h

#         # 1) Reshape everything you saved in the forward pass.
#         # 2) Compute all of the derivatives
#         # 3) Know that the autograders the gradients in a certain order, and the
#         #    local autograder will tell you which gradient you are currently failing.
#         d16 = delta
#         d15 = delta

#         d13 = d16 * self.z4
#         d4 = d16 * self.z13

#         d14 = d15 * self.hidden
#         dh = d15 * self.z14

#         d4 += -d14

#         d12 = d13 * (1 - self.h_act(self.z12) * self.h_act(self.z12)).T

#         d10 = d12
#         d11 = d12

#         self.dWnx += np.dot(d11.T, self.x.reshape(1,-1))
#         dx_t = np.dot(d11, self.Wnx)

#         self.dWnh += np.dot(d10.T, np.reshape(self.z9, (1, -1)))
#         d9 = np.dot(d10, self.Wnh)

#         d8 = d9 * self.hidden
#         dh += d9 * self.r

#         d7 = d8 * self.r_act(self.z7) * (1 - self.r_act(self.z7))

#         d5 = d7
#         d6 = d7

#         self.dWrx += np.dot(d6.T, np.reshape(self.x, (1, -1)))
#         dx_t += np.dot(d6, self.Wrx)

#         self.dWrh += np.dot(d5.T, np.reshape(self.hidden, (1, -1)))
#         dh += np.dot(d5, self.Wrh)

#         d3 = d4 * self.z_act(self.z3) * (1 - self.z_act(self.z3))

#         d2 = d3
#         d1 = d3

#         self.dWzx += np.dot(d2.T, np.reshape(self.x, (1, -1)))
#         dx_t += np.dot(d2, self.Wzx)

#         self.dWzh += np.dot(d1.T, np.reshape(self.hidden, (1, -1)))
#         dh += np.dot(d1, self.Wzh)


#         assert dx_t.shape == (1, self.d)
#         assert dh.shape == (1, self.h)

#         # return dx, dh
#         return dx_t, dh
