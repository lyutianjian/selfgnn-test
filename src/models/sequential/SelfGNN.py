# -*- coding: UTF-8 -*-

"""SelfGNN
Reference:
    Cross-view Collaborative Self-Attention Network for Sequential Recommendation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.BaseModel import SequentialModel
from utils import layers


class SelfGNN(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads', 'graph_num', 'gnn_layer', 'att_layer']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors (latdim).')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of GRU layers.')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of attention heads.')
        parser.add_argument('--dropout', type=float, default=0.1,
                            help='Dropout probability.')
        parser.add_argument('--graph_num', type=int, default=6,
                            help='Number of time-based graphs (time periods).')
        parser.add_argument('--time_periods', type=int, default=6,
                            help='Alias for graph_num (time periods).')
        parser.add_argument('--gnn_layer', type=int, default=2,
                            help='Number of GNN message passing layers.')
        parser.add_argument('--att_layer', type=int, default=2,
                            help='Number of self-attention layers for sequence.')
        parser.add_argument('--ssl_weight', type=float, default=0.000001,
                            help='Weight for SSL loss (ssl_reg).')
        parser.add_argument('--query_dim', type=int, default=64,
                            help='Query vector dimension for additive attention.')
        parser.add_argument('--pos_length', type=int, default=20,
                            help='Maximum position length for sequence.')
        parser.add_argument('--max_time', type=int, default=100,
                            help='Maximum time periods.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.dropout = args.dropout if hasattr(args, 'dropout') else 0.0
        # Support both --graph_num and --time_periods
        self.graph_num = args.time_periods if hasattr(args, 'time_periods') else args.graph_num
        self.gnn_layer = args.gnn_layer
        self.att_layer = args.att_layer
        self.ssl_weight = args.ssl_weight
        self.query_dim = args.query_dim
        self.pos_length = args.pos_length
        self.max_time = args.max_time if hasattr(args, 'max_time') else 100
        
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        # ========== Embedding Layers ==========
        # Multi-graph User and Item Embeddings
        # Original: uEmbed [graph_num, user, latdim]
        # Original: iEmbed [graph_num, item, latdim]
        self.user_embeddings = nn.ModuleList([
            nn.Embedding(self.user_num, self.emb_size) 
            for _ in range(self.graph_num)
        ])
        self.item_embeddings = nn.ModuleList([
            nn.Embedding(self.item_num, self.emb_size) 
            for _ in range(self.graph_num)
        ])
        
        # Position Embedding for sequence
        # Original: posEmbed [pos_length, latdim]
        self.pos_embeddings = nn.Embedding(self.pos_length, self.emb_size)
        
        # Time Embedding
        # Original: timeEmbed [maxTime+1, latdim]
        self.time_embeddings = nn.Embedding(self.max_time + 1, self.emb_size)
        
        # Item Embedding for attention-based sequence modeling
        # Original: iEmbed_att (final_item_vector after aggregation)
        # This will be computed dynamically, not a separate parameter
        
        # ========== GNN Message Passing Layers ==========
        # LightGCN style message propagation
        # Original code uses messagePropagate function with timeEmbed projection
        self.time_projections = nn.ModuleList([
            nn.Linear(self.emb_size, self.emb_size) 
            for _ in range(self.graph_num)
        ])
        
        # ========== Graph-level Sequential Modeling ==========
        # GRU for processing multi-graph embeddings
        # Original: BasicLSTMCell, but paper suggests GRU
        # Input: [batch, graph_num, emb_size] -> Output: [batch, graph_num, emb_size]
        self.graph_gru = nn.GRU(
            input_size=self.emb_size,
            hidden_size=self.emb_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # Multi-head Self-Attention for graph-level aggregation
        # Original: multihead_self_attention0 for users, multihead_self_attention1 for items
        self.user_graph_attention = layers.TransformerLayer(
            d_model=self.emb_size,
            d_ff=self.emb_size,
            n_heads=self.num_heads,
            dropout=self.dropout,
            kq_same=True
        )
        self.item_graph_attention = layers.TransformerLayer(
            d_model=self.emb_size,
            d_ff=self.emb_size,
            n_heads=self.num_heads,
            dropout=self.dropout,
            kq_same=True
        )
        
        # ========== Sequence-level Self-Attention ==========
        # Multi-head Self-Attention for item sequence
        # Original: multihead_self_attention_sequence (list of att_layer)
        self.sequence_attentions = nn.ModuleList([
            layers.TransformerLayer(
                d_model=self.emb_size,
                d_ff=self.emb_size,
                n_heads=self.num_heads,
                dropout=self.dropout,
                kq_same=False
            )
            for _ in range(self.att_layer)
        ])
        
        # ========== Personalized Weight Generation ==========
        # Meta network to generate personalized weights for each graph
        # Original: meta1 = concat[final_user * user_vector[i], final_user, user_vector[i]]
        #           meta2 = FC(meta1, ssldim, leakyRelu)
        #           weight = FC(meta2, 1, sigmoid)
        self.meta_fc1 = nn.Linear(self.emb_size * 3, self.emb_size)
        self.meta_fc2 = nn.Linear(self.emb_size, 1)
        
        # Layer Normalization
        self.user_ln = nn.LayerNorm(self.emb_size)
        self.item_ln = nn.LayerNorm(self.emb_size)
        self.seq_ln = nn.LayerNorm(self.emb_size)

    def forward(self, feed_dict):
        """
        Forward pass of SelfGNN
        
        Interface compliance with ReChorus:
        - Input: feed_dict with 'user_id', 'item_id', 'history_items', 'lengths'
        - Output: Dictionary with 'prediction' key containing [batch_size, n_candidates] scores
        - Compatible with SequentialModel.loss() which expects first column = positive, rest = negatives
        
        Args:
            feed_dict: Dictionary containing:
                - user_id: [batch_size]
                - item_id: [batch_size, -1] (target items + negatives)
                - history_items: [batch_size, history_max]
                - history_times: [batch_size, history_max] (optional)
                - lengths: [batch_size]
        
        Returns:
            Dictionary with 'prediction': [batch_size, n_candidates]
        """
        self.check_list = []
        
        user_ids = feed_dict['user_id']  # [batch_size]
        item_ids = feed_dict['item_id']  # [batch_size, -1]
        history_items = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]
        batch_size = user_ids.shape[0]
        n_candidates = item_ids.shape[1]
        
        # ========== Step 1: Short-term Encoding ==========
        # Build short-term graphs from history sequences
        edge_indices, edge_weights = self._build_short_term_graphs(history_items, lengths)
        
        # Multi-graph GNN message passing
        user_vectors = []  # List of [batch_size, emb_size] for each graph
        item_vectors = []  # List of [item_num, emb_size] for each graph
        
        for k in range(self.graph_num):
            # Initialize embeddings for this graph
            user_emb = self.user_embeddings[k](user_ids)  # [batch_size, emb_size]
            item_emb = self.item_embeddings[k].weight  # [item_num, emb_size]
            
            # Multi-layer LightGCN message propagation
            user_embs = [user_emb]
            item_embs = [item_emb]
            
            for layer in range(self.gnn_layer):
                # Message propagation on items (using the co-occurrence graph)
                # Add time embedding projection as in original code
                aggregated_item = self._message_propagate(
                    item_embs[-1], 
                    edge_indices[k], 
                    edge_weights[k], 
                    self.item_num,
                    graph_idx=k  # Pass graph index for time projection
                )
                
                # Lookup aggregated embeddings for users' history items
                # Simple aggregation: mean pooling over history
                valid_mask = (history_items > 0).float()  # [batch_size, max_seq_len]
                history_embs = aggregated_item[history_items]  # [batch_size, max_seq_len, emb_size]
                aggregated_user = (history_embs * valid_mask.unsqueeze(-1)).sum(1) / (valid_mask.sum(1, keepdim=True) + 1e-8)
                
                # Residual connection
                user_embs.append(aggregated_user + user_embs[-1])
                item_embs.append(aggregated_item + item_embs[-1])
            
            # Sum all layers (like LightGCN)
            final_user = torch.stack(user_embs, dim=0).sum(0)  # [batch_size, emb_size]
            final_item = torch.stack(item_embs, dim=0).sum(0)  # [item_num, emb_size]
            
            user_vectors.append(final_user)
            item_vectors.append(final_item)
        
        # Stack graph embeddings: [graph_num, batch_size, emb_size]
        user_vectors = torch.stack(user_vectors, dim=0)
        item_vectors = torch.stack(item_vectors, dim=0)
        
        # Store for SSL loss computation
        self.short_term_user_vectors = user_vectors  # [graph_num, batch_size, emb_size]
        self.short_term_item_vectors = item_vectors  # [graph_num, item_num, emb_size]
        
        # ========== Step 2: Interval-level Modeling (GRU) ==========
        # Transpose to [batch_size, graph_num, emb_size] for GRU
        user_vectors_t = user_vectors.permute(1, 0, 2)  # [batch_size, graph_num, emb_size]
        item_vectors_t = item_vectors.permute(1, 0, 2)  # [item_num, graph_num, emb_size]
        
        # Apply GRU to model temporal dependencies across graphs
        user_gru_out, _ = self.graph_gru(user_vectors_t)  # [batch_size, graph_num, emb_size]
        item_gru_out, _ = self.graph_gru(item_vectors_t)  # [item_num, graph_num, emb_size]
        
        # Apply layer normalization
        user_gru_out = self.user_ln(user_gru_out)
        item_gru_out = self.item_ln(item_gru_out)
        
        # Apply multi-head self-attention for graph-level aggregation
        user_att_out = self.user_graph_attention(user_gru_out)  # [batch_size, graph_num, emb_size]
        item_att_out = self.item_graph_attention(item_gru_out)  # [item_num, graph_num, emb_size]
        
        # Aggregate across graphs (mean pooling)
        final_user_vector = user_att_out.mean(dim=1)  # [batch_size, emb_size]
        final_item_vector = item_att_out.mean(dim=1)  # [item_num, emb_size]
        
        # Store for SSL loss computation
        self.final_user_vector = final_user_vector
        self.final_item_vector = final_item_vector
        
        # ========== Step 3: Instance-level Modeling (Sequence Attention) ==========
        # Get item embeddings from aggregated item vectors
        history_item_embs = final_item_vector[history_items]  # [batch_size, max_seq_len, emb_size]
        
        # Add position embeddings
        seq_len = history_items.shape[1]
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        positions = torch.clamp(positions, 0, self.pos_length - 1)
        pos_embs = self.pos_embeddings(positions)  # [batch_size, max_seq_len, emb_size]
        
        # Combine item and position embeddings
        sequence_input = self.seq_ln(history_item_embs + pos_embs)  # [batch_size, max_seq_len, emb_size]
        
        # Apply multi-layer self-attention
        valid_mask = (history_items > 0).float()  # [batch_size, max_seq_len]
        attn_mask = valid_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, max_seq_len]
        
        seq_output = sequence_input
        for att_layer in self.sequence_attentions:
            seq_output = att_layer(seq_output, attn_mask) + seq_output  # Residual connection
        
        # Aggregate sequence output (mean pooling over valid positions)
        seq_user_vector = (seq_output * valid_mask.unsqueeze(-1)).sum(1) / (valid_mask.sum(1, keepdim=True) + 1e-8)
        
        # ========== Step 4: Fusion and Final Prediction ==========
        # Combine interval-level and instance-level representations
        # Original code adds them together
        combined_user = final_user_vector + seq_user_vector  # [batch_size, emb_size]
        
        # Get target item embeddings
        target_item_embs = final_item_vector[item_ids]  # [batch_size, n_candidates, emb_size]
        
        # Compute prediction scores
        prediction = (combined_user.unsqueeze(1) * target_item_embs).sum(-1)  # [batch_size, n_candidates]
        
        return {'prediction': prediction}
    
    def _build_short_term_graphs(self, history_items, lengths):
        """
        Build short-term graphs by splitting history sequences into T time periods.
        Each time period forms a subgraph based on item co-occurrence within the batch.
        
        Note on sparse matrix handling:
        - Original TF code uses tf.sparse.SparseTensor for adjacency matrices
        - This PyTorch implementation uses edge_index format (similar to PyTorch Geometric)
        - Edge_index: [2, num_edges] where row 0 = source, row 1 = target
        - This is equivalent to COO (Coordinate) sparse format
        - Message propagation uses scatter_add instead of sparse matrix multiplication
        
        Args:
            history_items: [batch_size, max_seq_len] - User interaction sequences
            lengths: [batch_size] - Actual length of each sequence
        
        Returns:
            edge_indices: List of T edge_index tensors, each [2, num_edges_t]
            edge_weights: List of T edge weight tensors, each [num_edges_t]
        """
        batch_size, max_seq_len = history_items.shape
        device = history_items.device
        
        edge_indices = []
        edge_weights = []
        
        # Precompute position indices matrix [B, L]
        pos_idx = torch.arange(max_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        valid_item_mask = (history_items > 0)
        
        # Vectorized construction per time period
        for t in range(self.graph_num):
            # Period length per user (at least 1)
            period_len = torch.div(lengths, self.graph_num, rounding_mode='floor')
            period_len = torch.clamp(period_len, min=1)
            
            start = (period_len * t)
            # For last period, end = seq_len; else end = min((t+1)*period_len, seq_len)
            end_candidate = period_len * (t + 1)
            end = torch.minimum(end_candidate, lengths)
            end = torch.where(torch.tensor(t == self.graph_num - 1, device=device), lengths, end)
            
            # Mask positions within period and within actual sequence length
            period_mask = (pos_idx >= start.unsqueeze(1)) & (pos_idx < end.unsqueeze(1)) & (pos_idx < lengths.unsqueeze(1))
            period_valid_mask = period_mask & valid_item_mask
            
            # Self-loops: items within the period
            period_items = history_items[period_valid_mask]
            # Adjacent pairs within the same period
            if max_seq_len > 1:
                period_mask_i = period_mask[:, :-1]
                period_mask_j = period_mask[:, 1:]
                pair_mask = period_mask_i & period_mask_j & valid_item_mask[:, :-1] & valid_item_mask[:, 1:]
                src_pairs = history_items[:, :-1][pair_mask]
                tgt_pairs = history_items[:, 1:][pair_mask]
            else:
                src_pairs = torch.empty(0, dtype=torch.long, device=device)
                tgt_pairs = torch.empty(0, dtype=torch.long, device=device)
            
            # Combine self-loops and adjacent pairs
            src_all = torch.cat([period_items, src_pairs], dim=0)
            tgt_all = torch.cat([period_items, tgt_pairs], dim=0)
            
            if src_all.numel() == 0:
                edge_index = torch.zeros((2, 1), dtype=torch.long, device=device)
                edge_weight = torch.zeros(1, dtype=torch.float32, device=device)
            else:
                # Count edges via unique over linearized indices
                lin_idx = src_all * self.item_num + tgt_all
                uniq, counts = torch.unique(lin_idx, return_counts=True)
                src = torch.div(uniq, self.item_num, rounding_mode='floor')
                tgt = uniq % self.item_num
                edge_index = torch.stack([src, tgt], dim=0)
                edge_weight = counts.float()
                edge_weight = edge_weight / (edge_weight.max() + 1e-8)
            
            edge_indices.append(edge_index)
            edge_weights.append(edge_weight)
        
        return edge_indices, edge_weights
    
    def _message_propagate(self, item_embeddings, edge_index, edge_weight, num_items, graph_idx=0):
        """
        LightGCN-style message propagation on a graph.
        
        Args:
            item_embeddings: [num_items, emb_size] - Item embeddings for this graph
            edge_index: [2, num_edges] - Edge indices (source, target)
            edge_weight: [num_edges] - Edge weights (co-occurrence strength)
            num_items: Total number of items
            graph_idx: Index of current graph (unused, kept for compatibility)
        
        Returns:
            aggregated_embeddings: [num_items, emb_size] - Aggregated item embeddings
        """
        device = item_embeddings.device
        
        if edge_index.shape[1] == 0 or (edge_index == 0).all():
            # Empty graph - return zero embeddings
            return torch.zeros(num_items, self.emb_size, device=device)
        
        # Extract source and target nodes
        src_nodes = edge_index[0]  # [num_edges]
        tgt_nodes = edge_index[1]  # [num_edges]
        
        # Gather source embeddings
        src_embeddings = item_embeddings[src_nodes]  # [num_edges, emb_size]
        
        # Weight the embeddings by edge strength (co-occurrence count)
        weighted_embeddings = src_embeddings * edge_weight.unsqueeze(1)  # [num_edges, emb_size]
        
        # Aggregate to target nodes using scatter_add
        aggregated = torch.zeros(num_items, self.emb_size, device=device)
        aggregated.scatter_add_(0, tgt_nodes.unsqueeze(1).expand(-1, self.emb_size), 
                                weighted_embeddings)
        
        return aggregated
    
    def loss(self, out_dict: dict) -> torch.Tensor:
        """
        Compute total loss = recommendation loss + SSL loss
        
        Args:
            out_dict: Output from forward(), containing 'prediction'
        
        Returns:
            total_loss: Scalar tensor
        """
        # ========== Part 1: Recommendation Loss ==========
        # Use the standard BPR loss from parent class
        predictions = out_dict['prediction']  # [batch_size, n_candidates]
        batch_size = predictions.shape[0]
        n_candidates = predictions.shape[1]
        
        # Assuming first column is positive, rest are negative
        pos_pred = predictions[:, 0]  # [batch_size]
        neg_pred = predictions[:, 1:]  # [batch_size, n_neg]
        
        # BPR loss with softmax on negatives (from BaseModel)
        neg_softmax = (neg_pred - neg_pred.max(dim=1, keepdim=True)[0]).softmax(dim=1)
        rec_loss = -(((pos_pred.unsqueeze(1) - neg_pred).sigmoid() * neg_softmax).sum(dim=1)).clamp(min=1e-8, max=1-1e-8).log().mean()
        
        # ========== Part 2: SSL Loss ==========
        ssl_loss = self._compute_ssl_loss(batch_size)
        
        # ========== Total Loss ==========
        total_loss = rec_loss + self.ssl_weight * ssl_loss
        
        return total_loss
    
    def _compute_ssl_loss(self, batch_size):
        """
        Compute self-supervised learning loss for cross-view contrastive learning.
        
        Original TF logic:
        1. Generate personalized weights for each graph
        2. Sample positive and negative pairs within batch
        3. Compute long-term predictions (final vectors)
        4. Compute short-term predictions (graph-specific vectors)
        5. SSL loss = max(0, 1 - S_final * (posPred - negPred))
        
        Returns:
            ssl_loss: Scalar tensor
        """
        if not hasattr(self, 'short_term_user_vectors') or not hasattr(self, 'final_user_vector'):
            return torch.tensor(0.0, device=self.device)
        
        # ========== Step 1: Generate Personalized Weights ==========
        # meta1 = concat[final_user * user_vector[i], final_user, user_vector[i]]
        # user_weight[i] = sigmoid(FC(leaky_relu(FC(meta1))))
        
        user_weights = []  # List of [batch_size] for each graph
        
        for k in range(self.graph_num):
            short_term_user = self.short_term_user_vectors[k]  # [batch_size, emb_size]
            
            # Concatenate: element-wise product, final, short-term
            meta_input = torch.cat([
                self.final_user_vector * short_term_user,
                self.final_user_vector,
                short_term_user
            ], dim=-1)  # [batch_size, emb_size * 3]
            
            # Meta network: FC -> LeakyReLU -> FC -> Sigmoid
            meta_hidden = F.leaky_relu(self.meta_fc1(meta_input), negative_slope=0.2)
            weight = torch.sigmoid(self.meta_fc2(meta_hidden)).squeeze(-1)  # [batch_size]
            
            user_weights.append(weight)
        
        user_weights = torch.stack(user_weights, dim=0)  # [graph_num, batch_size]
        
        # ========== Step 2: Sample Positive and Negative Pairs ==========
        # Simplified approach: Use current batch as contrastive samples
        # For each user-item pair in batch, create negative samples by random sampling
        
        ssl_loss = 0.0
        num_samples = min(batch_size // 2, 32)  # Limit number of samples for efficiency
        
        if num_samples == 0:
            return torch.tensor(0.0, device=self.device)
        
        for k in range(self.graph_num):
            # Sample user-item pairs from batch
            # Positive pairs: random selection from batch
            pos_indices = torch.randperm(batch_size, device=self.device)[:num_samples]
            
            # Negative pairs: same users, but random items (different from positive)
            neg_indices = torch.randperm(batch_size, device=self.device)[:num_samples]
            
            # Get user and item indices
            pos_users = pos_indices
            neg_users = neg_indices
            
            # Sample random items for positive and negative
            pos_items = torch.randint(1, self.item_num, (num_samples,), device=self.device)
            neg_items = torch.randint(1, self.item_num, (num_samples,), device=self.device)
            
            # ========== Step 3: Compute Long-term Predictions ==========
            # Use final aggregated vectors
            pos_user_final = self.final_user_vector[pos_users]  # [num_samples, emb_size]
            neg_user_final = self.final_user_vector[neg_users]  # [num_samples, emb_size]
            
            pos_item_final = self.final_item_vector[pos_items]  # [num_samples, emb_size]
            neg_item_final = self.final_item_vector[neg_items]  # [num_samples, emb_size]
            
            # Predictions (inner product with activation)
            pos_pred_final = (F.leaky_relu(pos_user_final * pos_item_final, negative_slope=0.2)).sum(-1)
            neg_pred_final = (F.leaky_relu(neg_user_final * neg_item_final, negative_slope=0.2)).sum(-1)
            
            # Get personalized weights
            pos_weight = user_weights[k, pos_users]
            neg_weight = user_weights[k, neg_users]
            
            # Weighted difference (stop gradient on predictions)
            S_final = pos_weight * pos_pred_final.detach() - neg_weight * neg_pred_final.detach()
            
            # ========== Step 4: Compute Short-term Predictions ==========
            # Use graph-specific vectors
            pos_user_short = self.short_term_user_vectors[k, pos_users]  # [num_samples, emb_size]
            neg_user_short = self.short_term_user_vectors[k, neg_users]  # [num_samples, emb_size]
            
            pos_item_short = self.short_term_item_vectors[k, pos_items]  # [num_samples, emb_size]
            neg_item_short = self.short_term_item_vectors[k, neg_items]  # [num_samples, emb_size]
            
            # Predictions
            pos_pred_short = (F.leaky_relu(pos_user_short * pos_item_short, negative_slope=0.2)).sum(-1)
            neg_pred_short = (F.leaky_relu(neg_user_short * neg_item_short, negative_slope=0.2)).sum(-1)
            
            # ========== Step 5: SSL Loss ==========
            # Original formula: max(0, 1 - S_final * (posPred - negPred))
            loss_k = F.relu(1.0 - S_final * (pos_pred_short - neg_pred_short))
            ssl_loss = ssl_loss + loss_k.sum()
        
        return ssl_loss / (self.graph_num * num_samples + 1e-8)
