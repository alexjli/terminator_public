from struct2seq.protein_features import *
from .term_attn import gather_nodes

class IndexDiffEncoding(PositionalEncodings):
    def __init__(self, num_embeddings, period_range=[2,1000], inf = 10000):
        super(IndexDiffEncoding, self).__init__(num_embeddings, period_range)
        self.inf = inf

    def forward(self, E_idx, chain_idx):
        dev = E_idx.device
        # i-j
        N_batch = E_idx.size(0)
        N_terms = E_idx.size(1)
        N_nodes = E_idx.size(2)
        N_neighbors = E_idx.size(3)
        ii = torch.arange(N_nodes, dtype=torch.float32).view((1, -1, 1)).to(dev)
        d_raw = (E_idx.float() - ii)

        chain_idx_expand = chain_idx.view(1, 1, -1, 1).expand((N_batch, N_terms, -1, N_neighbors))
        E_chain_idx = torch.gather(chain_idx_expand, 2, E_idx)
        same_chain = (E_chain_idx == E_chain_idx[:, :, :, 0:1]).to(dev)
        d = torch.where(same_chain, d_raw, torch.tensor(self.inf).float().to(dev)).unsqueeze(-1)

        print(d.long().squeeze(-1))
        # Original Transformer frequencies
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32)
            * -(np.log(10000.0) / self.num_embeddings)
        ).to(dev)
        # Grid-aligned
        # frequency = 2. * np.pi * torch.exp(
        #     -torch.linspace(
        #         np.log(self.period_range[0]),
        #         np.log(self.period_range[1]),
        #         self.num_embeddings / 2
        #     )
        # )
        angles = d * frequency.view((1,1,1,-1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

class TERMProteinFeatures(ProteinFeatures):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, features_type='full', augment_eps=0., dropout=0.1):
        """ variant of struct2seq protein features, but with batches of TERMs as sequences """
        super(TERMProteinFeatures, self).__init__(edge_features, node_features, 
        num_positional_embeddings=16, num_rbf=16, top_k=30, features_type='full', 
        augment_eps=0., dropout=0.1)

        self.embeddings = IndexDiffEncoding(num_positional_embeddings)
        
    def _dist(self, X, mask, eps=1E-6):
        """ Pairwise euclidean distances """
        N_batch, N_terms, N_nodes, _ = X.shape

        # Convolutional network on NCHW
        mask_2D = torch.unsqueeze(mask,2) * torch.unsqueeze(mask,3)
        dX = torch.unsqueeze(X,2) - torch.unsqueeze(X,3)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 4) + eps)

        # Order TERM nodes by increasing distance (including self)
        # we add 1 to D_max since the D_max is the largest distance within a TERM
        # but since the length of the TERM is most often less than the dimension of
        # the matrix, we get weird behavior with topk later using just D_max
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_max += 1
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, N_nodes, dim=-1, largest=False)

        # Debug plot KNN
        # print(E_idx[:10,:10])
        # D_simple = mask_2D * torch.zeros(D.size()).scatter(-1, E_idx, torch.ones_like(knn_D))
        # print(D_simple)
        # fig = plt.figure(figsize=(4,4))
        # ax = fig.add_subplot(111)
        # D_simple = D.data.numpy()[0,:,:]
        # plt.imshow(D_simple, aspect='equal')
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig('D_knn.pdf')
        # exit(0)
        return D_neighbors, E_idx

    def _rbf(self, D):
        dev = D.device
        # Distance radial basis function
        D_min, D_max, D_count = 0., 20., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count).to(dev)
        D_mu = D_mu.view([1,1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)

        # for i in range(D_count):
        #     fig = plt.figure(figsize=(4,4))
        #     ax = fig.add_subplot(111)
        #     rbf_i = RBF.data.numpy()[0,i,:,:]
        #     # rbf_i = D.data.numpy()[0,0,:,:]
        #     plt.imshow(rbf_i, aspect='equal')
        #     plt.axis('off')
        #     plt.tight_layout()
        #     plt.savefig('rbf{}.pdf'.format(i))
        #     print(np.min(rbf_i), np.max(rbf_i), np.mean(rbf_i))
        # exit(0)
        return RBF

    def _quaternions(self, R):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz,
            - Rxx + Ryy - Rzz,
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i,j: R[:,:,:,:,i,j]
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2),
            _R(0,2) - _R(2,0),
            _R(1,0) - _R(0,1)
        ], -1))
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)

        # Axis of rotation
        # Replace bad rotation matrices with identity
        # I = torch.eye(3).view((1,1,1,3,3))
        # I = I.expand(*(list(R.shape[:3]) + [-1,-1]))
        # det = (
        #     R[:,:,:,0,0] * (R[:,:,:,1,1] * R[:,:,:,2,2] - R[:,:,:,1,2] * R[:,:,:,2,1])
        #     - R[:,:,:,0,1] * (R[:,:,:,1,0] * R[:,:,:,2,2] - R[:,:,:,1,2] * R[:,:,:,2,0])
        #     + R[:,:,:,0,2] * (R[:,:,:,1,0] * R[:,:,:,2,1] - R[:,:,:,1,1] * R[:,:,:,2,0])
        # )
        # det_mask = torch.abs(det.unsqueeze(-1).unsqueeze(-1))
        # R = det_mask * R + (1 - det_mask) * I

        # DEBUG
        # https://math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        # Columns of this are in rotation plane
        # A = R - I
        # v1, v2 = A[:,:,:,:,0], A[:,:,:,:,1]
        # axis = F.normalize(torch.cross(v1, v2), dim=-1)
        return Q

    def _orientations_coarse(self, X, E_idx, eps=1e-6):
        # Pair features

        # Shifted slices of unit vectors
        dX = X[:,:,1:,:] - X[:,:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:,:-2,:]
        u_1 = U[:,:,1:-1,:]
        u_0 = U[:,:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Bond angle calculation
        cosA = -(u_1 * u_0).sum(-1)
        cosA = torch.clamp(cosA, -1+eps, 1-eps)
        A = torch.acos(cosA)
        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # Build relative orientations
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 3)
        O = O.view(list(O.shape[:3]) + [9])
        O = F.pad(O, (0,0,1,2), 'constant', 0)

        # DEBUG: Viz [dense] pairwise orientations
        # O = O.view(list(O.shape[:2]) + [3,3])
        # dX = X.unsqueeze(2) - X.unsqueeze(1)
        # dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        # dU = dU / torch.norm(dU, dim=-1, keepdim=True)
        # dU = (dU + 1.) / 2.
        # plt.imshow(dU.data.numpy()[0])
        # plt.show()
        # print(dX.size(), O.size(), dU.size())
        # exit(0)

        O_neighbors = gather_nodes(O, E_idx)
        X_neighbors = gather_nodes(X, E_idx)

        # Re-view as rotation matrices
        O = O.view(list(O.shape[:3]) + [3,3])
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:4]) + [3,3])

        # Rotate into local reference frames
        dX = X_neighbors - X.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(3), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(3).transpose(-1,-2), O_neighbors)
        Q = self._quaternions(R)

        # Orientation features
        O_features = torch.cat((dU,Q), dim=-1)

        # DEBUG: Viz pairwise orientations
        # IMG = Q[:,:,:,:3]
        # # IMG = dU
        # dU_full = torch.zeros(X.shape[0], X.shape[1], X.shape[1], 3).scatter(
        #     2, E_idx.unsqueeze(-1).expand(-1,-1,-1,3), IMG
        # )
        # print(dU_full)
        # dU_full = (dU_full + 1.) / 2.
        # plt.imshow(dU_full.data.numpy()[0])
        # plt.show()
        # exit(0)
        # print(Q.sum(), dU.sum(), R.sum())
        return O_features

    def _dihedrals(self, X, eps=1e-7):
        # First 3 coordinates are N, CA, C
        X = X[:,:,:,:3,:].reshape(X.shape[0], X.shape[1], 3*X.shape[2], 3)

        # Shifted slices of unit vectors
        dX = X[:,:,1:,:] - X[:,:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:,:-2,:]
        u_1 = U[:,:,1:-1,:]
        u_0 = U[:,:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1,2), 'constant', 0)
        D = D.view((D.size(0), D.size(1), int(D.size(2)/3), 3))
        phi, psi, omega = torch.unbind(D,-1)

        # print(cosD.cpu().data.numpy().flatten())
        # print(omega.sum().cpu().data.numpy().flatten())

        # Bond angle calculation
        # A = torch.acos(-(u_1 * u_0).sum(-1))

        # DEBUG: Ramachandran plot
        # x = phi.cpu().data.numpy().flatten()
        # y = psi.cpu().data.numpy().flatten()
        # plt.scatter(x * 180 / np.pi, y * 180 / np.pi, s=1, marker='.')
        # plt.xlabel('phi')
        # plt.ylabel('psi')
        # plt.axis('square')
        # plt.grid()
        # plt.axis([-180,180,-180,180])
        # plt.show()

        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 3)
        return D_features

    def forward(self, X, batched_focuses, chain_idx, mask):
        """ Featurize coordinates as an attributed graph """

        # Data augmentation
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        # Build edges in order of distance
        X_ca = X[:,:,:,1,:]
        D_neighbors, relative_E_idx = self._dist(X_ca, mask)

        # Pairwise features
        O_features = self._orientations_coarse(X_ca, relative_E_idx)
        RBF = self._rbf(D_neighbors)

        # E_idx initially only contains relative indices
        # convert to absolute indices using batched_focuses for pairwise embeddings
        N_nodes = relative_E_idx.size(2)
        absolute_E_idx = torch.gather(batched_focuses.unsqueeze(-2).expand(-1, -1, N_nodes, -1), -1, relative_E_idx)
        # Pairwise embeddings
        E_positional = self.embeddings(absolute_E_idx, chain_idx)

        # Full backbone angles
        V = self._dihedrals(X)
        E = torch.cat((E_positional, RBF, O_features), -1)
        
        # Embed the nodes
        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        
        # DEBUG
        # U = (np.nan * torch.zeros(X.size(0),X.size(1),X.size(1),3)).scatter(2, E_idx.unsqueeze(-1).expand(-1,-1,-1,3), E[:,:,:,:3])
        # plt.imshow(U.data.numpy()[0,:,:,0])
        # plt.show()
        # exit(0)
        return V, E, relative_E_idx
