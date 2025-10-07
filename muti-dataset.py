import os
import pdb
import numpy as np
import torch as th
import dgl
from dgl.data import (
    CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset,
    AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset,
    CoauthorCSDataset, CoauthorPhysicsDataset
)
from ogb.nodeproppred import DglNodePropPredDataset


def load(name, multimodal=False, feature_type='t5vit', multimodal_path=None):
    """
    Load dataset, supporting both single-modal and multi-modal datasets.

    Args:
        name (str): Dataset name (e.g., 'cora', 'citeseer', 'books-nc').
        multimodal (bool): Whether to load a multi-modal dataset.
        feature_type (str): Type of feature to load for multi-modal datasets.
        multimodal_path (str): Path to the multi-modal dataset.

    Returns:
        tuple: Graph, features (or text/vision features for multimodal), labels, number of classes,
               and train/val/test indices.
    """
    if multimodal:
        # Ensure the multimodal path is provided
        assert multimodal_path is not None, "Path to multimodal data must be provided."

        # Define file paths
        # import ipdb;ipdb.set_trace()
        edges_path = os.path.join(multimodal_path, "nc_edges-nodeid.pt")
        text_feat_path = os.path.join(multimodal_path, f"{feature_type}_feat.pt")  # Text features
        vision_feat_path = os.path.join(multimodal_path, f"{feature_type}_feat.pt")  # Vision features
        labels_path = os.path.join(multimodal_path, "labels-w-missing.pt")
        split_path = os.path.join(multimodal_path, "split.pt")

        # Ensure required files exist
        for path in [edges_path, text_feat_path, vision_feat_path, labels_path, split_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file {path} not found in {multimodal_path}.")

        # # Load edges
        # edges = th.load(edges_path)  # Format: (src, dst)
        # print(f"edges: {edges}")
        # src, dst = edges[0], edges[1]
        # graph = dgl.graph((src, dst))
        # Load edges
        # import ipdb;ipdb.set_trace()
        edges = th.load(edges_path)  # Expected format: (src, dst) or Nx2 tensor
        list1=[]
        list2=[]
        for i in  range (len(edges)):
            list1.append(edges[i][0])
            list2.append(edges[i][1])
        src, dst = th.tensor(list1), th.tensor(list2)
        # # Parse edges
        # if isinstance(edges, list) and len(edges) == 2:
        #     src, dst = edges
        #     src, dst = th.tensor(src), th.tensor(dst)
        # elif isinstance(edges, th.Tensor) and edges.dim() == 2:  # If edges are saved as Nx2 tensor
        #     src, dst = edges[:, 0], edges[:, 1]
        # else:
        #     raise ValueError("Unsupported edge format in file.")
        # Create graph
        graph = dgl.graph((src, dst))

        # Debug output
        print(f"Graph created with {graph.num_nodes()} nodes and {graph.num_edges()} edges.")


        #import ipdb;ipdb.set_trace()
        # Load features, labels, and splits
        text_feat = th.load(text_feat_path)
        # vision_feat = th.load(vision_feat_path)
        labels = th.load(labels_path)
        splits = th.load(split_path)

        # Ensure labels are Tensor
        if not isinstance(labels, th.Tensor):
            labels = th.tensor(labels)

        train_idx = splits['train_idx']
        val_idx = splits['val_idx']
        test_idx = splits['test_idx']

        
        # Compute number of classes
        #import ipdb;ipdb.set_trace()
        num_class = len(th.unique(labels))
     

        # print("===== Multimodal Dataset Loaded =====")
        # print(f"Graph: {graph}")
        # print(f"Text Features Shape: {text_feat.shape}")
        # print(f"Labels Shape: {labels.shape}")
        # print(f"Number of Classes: {num_class}")
        # print(f"Train Indices (first 5): {train_idx}")
        # print(f"Validation Indices (first 5): {val_idx}")
        # print(f"Test Indices (first 5): {test_idx}")

        return graph, text_feat, labels, num_class+1, train_idx, val_idx, test_idx




