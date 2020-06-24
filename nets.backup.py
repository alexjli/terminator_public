class CondenseMSA(nn.Module):
    def __init__(self, hidden_dim = 64, num_features = num_features, filter_len = 3, num_blocks = 6, nheads = 8, device = 'cuda:0'):
        super(CondenseMSA, self).__init__()
        channels = hidden_dim
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.embedding = ResidueFeatures(hidden_dim = hidden_dim, num_features = num_features)
        self.fe = FocusEncoding(hidden_dim = self.hidden_dim, dropout = 0.1, max_len = 1000)
        self.resnet = Conv1DResNet(filter_len = filter_len, channels = channels, num_blocks = num_blocks)
        self.transformer = TransformerEncoderLayer(d_model = hidden_dim, nhead = nheads, dim_feedforward = hidden_dim)
        self.encoder = nn.TransformerEncoder(self.transformer, num_layers=4)
        if torch.cuda.is_available():
            self.dev = device
        else:
            self.dev = 'cpu'
    """
    Only designed to handle one protein at a time
    I have no idea how to make this batchable
    """
    def forward0(self, X, features, term_lens, seq_len, focuses):
        import time
        last_timepoint = time.time()

        # embed MSAs and concat other features on
        embeddings = self.embedding(X, features)

        current_timepoint = time.time()
        print('embed', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # use Convolutional ResNet and averaging for further embedding and to reduce dimensionality
        convolution = self.resnet(embeddings)

        current_timepoint = time.time()
        print('convolve', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # split tensor into terms
        splits = torch.split(convolution, term_lens, dim = 1)

        # pad each term to the max term len and create its associated mask
        max_term_len = max(term_lens)
        padded_terms = []
        masks = []
        for term in splits:
            # term: 1 x TERM length x hidden dim
            term_len = term.shape[1]
            pad_len = max_term_len - term_len
            # make padding and cat it to the end of the term
            padding = torch.zeros((1, pad_len, self.hidden_dim)).to(self.dev)
            padded_t = torch.cat((term, padding), dim = 1)
            padded_terms.append(padded_t)

            # make mask
            mask = torch.ones(pad_len)
            unmask = torch.zeros(term_len)
            mask = torch.cat((unmask, mask)).bool()
            masks.append(mask)

        # create big batch of terms
        batch_terms = torch.cat(padded_terms, dim=0)
        #print(batch_terms.shape)
        # mask padded 0s
        src_key_mask = torch.stack(masks).to(self.dev)

        current_timepoint = time.time()
        print('batch', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # transpose because pytorch transformer uses weird shape
        batch_terms = batch_terms.transpose(0,1)
        # create node embeddings
        node_embeddings = self.encoder(batch_terms, src_key_padding_mask = src_key_mask)
        # transpose back
        node_embeddings = node_embeddings.transpose(0,1)

        current_timepoint = time.time()
        print('transform', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # create a space to aggregate term data
        aggregate = torch.zeros((seq_len, self.hidden_dim)).to(self.dev)
        #print(aggregate.shape)
        count = torch.zeros((seq_len,1)).to(self.dev)

        for i in range(len(splits)):
            # look at the unpadded term
            length = term_lens[i]
            term_n_embed = node_embeddings[i, :length, :]
            #print(term.shape)
            focus = focuses[i]
            #print(focus)

            # add term to aggregate at proper index
            aggregate[focus] += term_n_embed
            count[focus] += 1

        current_timepoint = time.time()
        print('aggregate', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # average the aggregate
        aggregate /= count
        aggregate.unsqueeze(0)

        return aggregate

    """
    forward0, but it uses built-in padding features to speed things up
    """
    def forward1(self, X, features, term_lens, seq_len, focuses):
        import time
        last_timepoint = time.time()

        # embed MSAs and concat other features on
        embeddings = self.embedding(X, features)

        current_timepoint = time.time()
        print('embed', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # use Convolutional ResNet and averaging for further embedding and to reduce dimensionality
        convolution = self.resnet(embeddings)

        current_timepoint = time.time()
        print('convolve', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # split tensor into terms
        splits = torch.split(convolution, term_lens, dim = 1)

        # pad each term to the max term len
        s_splits = [split.squeeze(0) for split in splits]
        batch_terms = pad_sequence(s_splits, batch_first = True)
        print(batch_terms.shape)

        current_timepoint = time.time()
        print('batch', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        zeros = torch.zeros(sum(term_lens)).to(self.dev)
        unmask = torch.split(zeros, term_lens)
        src_key_mask = pad_sequence(unmask, batch_first=True, padding_value=1).bool()

        #src_key_mask = src_key_mask.to(self.dev)

        current_timepoint = time.time()
        print('mask', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # transpose because pytorch transformer uses weird shape
        batch_terms = batch_terms.transpose(0,1)
        # create node embeddings
        node_embeddings = self.encoder(batch_terms, src_key_padding_mask = src_key_mask)
        # transpose back
        node_embeddings = node_embeddings.transpose(0,1)

        current_timepoint = time.time()
        print('transform', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # create a space to aggregate term data
        aggregate = torch.zeros((seq_len, self.hidden_dim)).to(self.dev)
        #print(aggregate.shape)
        count = torch.zeros((seq_len,1)).to(self.dev)

        for i in range(len(splits)):
            # look at the unpadded term
            length = term_lens[i]
            term_n_embed = node_embeddings[i, :length, :]
            #print(term.shape)
            focus = focuses[i]
            #print(focus)

            # add term to aggregate at proper index
            aggregate[focus] += term_n_embed
            count[focus] += 1

        current_timepoint = time.time()
        print('aggregate', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # average the aggregate
        aggregate /= count
        aggregate.unsqueeze(0)

        return aggregate

    """
    Use better masking to reduce dimensionality
    """
    def forward2(self, X, features, term_lens, seq_len, focuses):
        import time
        last_timepoint = time.time()

        # embed MSAs and concat other features on
        embeddings = self.embedding(X, features)

        current_timepoint = time.time()
        print('embed', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # use Convolutional ResNet and averaging for further embedding and to reduce dimensionality
        convolution = self.resnet(embeddings)

        current_timepoint = time.time()
        print('convolve', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # create source mask so that terms can attend to themselves but not each other
        # create blocks of interaction
        blocks = [np.ones((i,i)) for i in term_lens]
        # create a block diagonal matrix mask
        src_mask = block_diag(*blocks)
        # convert to bool tensor
        src_mask = torch.from_numpy(src_mask).bool()
        # invert, because 1 = mask and 0 = unmask
        src_mask = ~src_mask.to(self.dev)
        #print(src_mask)

        current_timepoint = time.time()
        print('mask', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # transpose because pytorch transformer uses weird shape
        batch_terms = convolution.transpose(0,1)
        # create node embeddings
        node_embeddings = self.encoder(batch_terms, mask = src_mask)
        # transpose back
        node_embeddings = node_embeddings.transpose(0,1)

        current_timepoint = time.time()
        print('transform', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # create a space to aggregate term data
        aggregate = torch.zeros((seq_len, self.hidden_dim)).to(self.dev)
        count = torch.zeros((seq_len,1)).to(self.dev)

        # flatten the focuses into one list, then a tensor
        flat_focus = []
        for l in focuses:
            flat_focus += l
        flat_focus = torch.Tensor(flat_focus).long().to(self.dev)

        # aggregate node embeddings and associated counts
        aggregate = aggregate.index_put((flat_focus,), node_embeddings, accumulate=True)
        count_idx = torch.ones_like(flat_focus).unsqueeze(-1).float()
        count = count.index_put((flat_focus,), count_idx, accumulate=True)

        current_timepoint = time.time()
        print('aggregate', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # average the aggregate
        aggregate /= count
        aggregate.unsqueeze(0)

        return aggregate

    """
    Wow, batching! But I should move some of the computation to preprocessing to speed up training
    """
    def forward3(self, X, features, term_lens, seq_lens, focuses):
        import time
        last_timepoint = time.time()

        # embed MSAs and concat other features on
        embeddings = self.embedding(X, features)

        current_timepoint = time.time()
        print('embed', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # use Convolutional ResNet and averaging for further embedding and to reduce dimensionality
        convolution = self.resnet(embeddings)

        current_timepoint = time.time()
        print('convolve', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # create source mask so that terms can attend to themselves but not each other
        # create blocks of interaction
        blocks = [np.ones((i,i)) for i in term_lens]
        # create a block diagonal matrix mask
        src_mask = block_diag(*blocks)
        # convert to bool tensor
        src_mask = torch.from_numpy(src_mask).bool()
        # invert, because 1 = mask and 0 = unmask
        src_mask = ~src_mask.to(self.dev)

        src_mask = torch.stack(2*[src_mask for _ in range(self.nheads)])

        current_timepoint = time.time()
        print('mask', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        convolution = torch.cat((convolution, convolution), dim=0)
        # transpose because pytorch transformer uses weird shape
        batch_terms = convolution.transpose(0,1)
        # create node embeddings
        node_embeddings = self.encoder(batch_terms, mask = src_mask)
        # transpose back
        node_embeddings = node_embeddings.transpose(0,1)

        current_timepoint = time.time()
        print('transform', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # create a space to aggregate term data
        aggregate = torch.zeros((2,seq_len, self.hidden_dim)).to(self.dev)
        count = torch.zeros((2,seq_len,1)).to(self.dev)

        # flatten the focuses into one list, then a tensor
        flat_focus = []
        for l in focuses:
            flat_focus += l
        flat_focus = torch.Tensor(flat_focus).long().to(self.dev)
        flat_focus = flat_focus.repeat(2,1)

        layer = torch.arange(0,2).unsqueeze(-1).to(self.dev)

        # aggregate node embeddings and associated counts
        aggregate = aggregate.index_put((layer, flat_focus), node_embeddings, accumulate=True)
        count_idx = torch.ones_like(flat_focus).unsqueeze(-1).float()
        count = count.index_put((layer,flat_focus), count_idx, accumulate=True)

        current_timepoint = time.time()
        print('aggregate', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # average the aggregate
        aggregate /= count
        aggregate.unsqueeze(0)

        return aggregate

    '''
    S p e e d
    Fully batched
    Monster data hog tho bc of large masks and attention
    '''
    def forward4(self, X, features, seq_lens, focuses, src_mask, src_key_mask):
        n_batches = X.shape[0]
        max_seq_len = max(seq_lens)
        import time
        last_timepoint = time.time()

        # use source mask so that terms can attend to themselves but not each
        # for more efficient computation, generate the mask over all heads first
        # first, stack the current source mask nhead times, and transpose to get batches of the same mask
        print(size(src_mask))
        src_mask = src_mask.unsqueeze(0).expand(self.nheads,-1,-1,-1).transpose(0,1)
        print(size(src_mask))
        # next, flatten the 0th dim to generate a 3d tensor
        dim = focuses.shape[1]
        src_mask = src_mask.contiguous()
        src_mask = src_mask.view(-1, dim, dim)
        #src_mask = torch.flatten(src_mask, 0, 1)
        print(size(src_mask))

        # zero out all positions used as padding so they don't contribute to aggregation
        negate_padding_mask = (~src_key_mask).unsqueeze(-1).expand(-1,-1, self.hidden_dim)

        current_timepoint = time.time()
        print('mask', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # embed MSAs and concat other features on
        embeddings = self.embedding(X, features)
        print(size(embeddings))

        current_timepoint = time.time()
        print('embed', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # use Convolutional ResNet and averaging for further embedding and to reduce dimensionality
        convolution = self.resnet(embeddings)
        # zero out biases introduced into padding
        convolution *= negate_padding_mask
        print(size(convolution))

        current_timepoint = time.time()
        print('convolve', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint


        # add absolute positional encodings before transformer
        batch_terms = self.fe(convolution, focuses)
        # transpose because pytorch transformer uses weird shape
        batch_terms = batch_terms.transpose(0,1)
        # create node embeddings
        node_embeddings = self.encoder(batch_terms, mask = src_mask, src_key_padding_mask = src_key_mask)
        # transpose back
        node_embeddings = node_embeddings.transpose(0,1)
        print(size(node_embeddings))

        # zero out biases introduced into padding
        node_embeddings *= negate_padding_mask

        current_timepoint = time.time()
        print('transform', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # create a space to aggregate term data
        aggregate = torch.zeros((n_batches, max_seq_len, self.hidden_dim)).to(self.dev).double()
        count = torch.zeros((n_batches, max_seq_len, 1)).to(self.dev).long()

        # this make sure each batch stays in the same layer during aggregation
        layer = torch.arange(n_batches).unsqueeze(-1).to(self.dev)

        # aggregate node embeddings and associated counts
        aggregate = aggregate.index_put((layer, focuses), node_embeddings, accumulate=True)
        count_idx = torch.ones_like(focuses).unsqueeze(-1).to(self.dev)
        count = count.index_put((layer, focuses), count_idx, accumulate=True)

        current_timepoint = time.time()
        print('aggregate', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # average the aggregate
        aggregate /= count

        return aggregate
