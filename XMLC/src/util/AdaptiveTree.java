package util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.SortedSet;
import java.util.stream.Collectors;

public class AdaptiveTree extends Tree {
	TreeNode root;
	Map<Integer, TreeNode> indexToNode = new HashMap<Integer, TreeNode>();
	Map<Integer, Integer> labelToIndex = new HashMap<Integer, Integer>();

	public AdaptiveTree() {
	}

	public AdaptiveTree(Tree tree, String treeType, boolean shuffleLabels) throws Exception {
		this(tree, treeType, shuffleLabels, null);
	}

	public AdaptiveTree(Tree tree, String treeType, SortedSet<Integer> labels) throws Exception {
		this(tree, treeType, false, labels);
	}

	private AdaptiveTree(Tree tree, String treeType, boolean shuffleLabels, SortedSet<Integer> labels)
			throws Exception {

		if (labels != null && !labels.isEmpty() && labels.size() != tree.m)
			throw new Exception("Size of labels set (" + labels.size()
					+ ") provided does not match to the number of leaves present in the tree (" + tree.m
					+ ")");

		// Get the basic details
		this.m = tree.m;
		this.k = tree.k;
		this.size = tree.size;
		this.numberOfInternalNodes = tree.numberOfInternalNodes;

		switch (treeType) {
		case CompleteTree.name:
			populateFromCompleteTree((CompleteTree) tree, labels);
			break;
		default:
			throw new Exception(String.format("AdaptiveTree for %s is not yet implemented.", treeType));
		}

		if (shuffleLabels)
			shuffleLabels();

	}

	private void shuffleLabels() {
		List<Integer> indices = new ArrayList<Integer>(labelToIndex.values());
		Collections.shuffle(indices);

		int index = 0;
		for (Entry<Integer, Integer> entry : labelToIndex.entrySet()) {
			Integer nodeIndex = indices.get(index);
			entry.setValue(nodeIndex);
			indexToNode.get(nodeIndex).label = entry.getKey();
			index++;
		}

	}

	@Override
	public ArrayList<Integer> getChildNodes(int node) {
		TreeNode treeNode = indexToNode.get(node);
		return treeNode.isLeaf()
				? null
				: treeNode.children.stream()
						.map(child -> child.index)
						.collect(Collectors.toCollection(ArrayList<Integer>::new));
	}

	@Override
	public int getParent(int node) {
		TreeNode treeNode = indexToNode.get(node);
		return treeNode.getParent() == null ? -1 : treeNode.getParent().index;
	}

	@Override
	public boolean isLeaf(int node) {
		TreeNode treeNode = indexToNode.get(node);
		return treeNode.isLeaf();
	}

	@Override
	public int getTreeIndex(int label) {
		return labelToIndex.containsKey(label) ? labelToIndex.get(label) : -1;
	}

	@Override
	public int getLabelIndex(int nodeIndex) {
		return this.indexToNode.get(nodeIndex).label;
	}

	public int getNodeDepth(int nodeIndex) {
		return indexToNode.get(nodeIndex)
				.getDepth();
	}

	public int getTreeDepth() {
		return computeTreeDepth(root);
	}

	private int computeTreeDepth(TreeNode node) {
		if (node.children.isEmpty())
			return node.getDepth();
		else
			return node.children.parallelStream()
					.map(child -> computeTreeDepth(child))
					.max(Integer::compare)
					.get();
	}

	public Set<Integer> getAllLabels() {
		return labelToIndex.keySet();
	}

	public int adaptLeaf(int lableIndex, int newLabel) {
		TreeNode leaf = indexToNode.get(labelToIndex.get(lableIndex));
		TreeNode parent = leaf.getParent();

		int maxTreeIndex = indexToNode.keySet()
				.stream()
				.max(Integer::compare)
				.get();

		maxTreeIndex++;
		TreeNode newLeaf = new TreeNode(maxTreeIndex, null, newLabel);
		indexToNode.put(maxTreeIndex, newLeaf);
		labelToIndex.put(newLabel, maxTreeIndex);

		size++;
		m++;

		if (parent != null && parent.children.size() < k) {
			newLeaf.setParent(parent);
		} else {
			maxTreeIndex++;
			TreeNode newParent = new TreeNode(maxTreeIndex, parent);
			indexToNode.put(maxTreeIndex, newParent);

			leaf.setParent(newParent);
			newLeaf.setParent(newParent);

			size++;
			numberOfInternalNodes++;

			/*a special case, when the tree starts with only one node (being both root and leaf at the same time)*/
			if (newParent.getDepth() == 1)
				root = newParent;
		}
		return newLeaf.index;
	}

	private void populateFromCompleteTree(CompleteTree tree, SortedSet<Integer> labels) throws Exception {
		if (tree != null) {
			// Start with root (for complete tree it is 0).
			int treeIndex = 0;
			this.root = new TreeNode(treeIndex);
			indexToNode.put(treeIndex, this.root);

			List<Integer> nodeList = new ArrayList<Integer>();
			nodeList.add(treeIndex);

			boolean labelsProvided = labels != null && !labels.isEmpty();
			Integer[] labelArr = new Integer[labelsProvided ? labels.size() : 0];
			if (labelsProvided)
				labels.toArray(labelArr);

			while (!nodeList.isEmpty()) {
				treeIndex = nodeList.remove(0);
				TreeNode parent = indexToNode.get(treeIndex);

				ArrayList<Integer> childNodes = tree.getChildNodes(treeIndex);

				if (childNodes != null)
					childNodes
							.stream()
							.forEach(nodeIndex -> {

								TreeNode child = new TreeNode(nodeIndex, parent);

								if (tree.isLeaf(nodeIndex)) {
									if (labelsProvided)
										manageNodeLabel(child, nodeIndex, labelArr[labelToIndex.size()]);
									else
										manageNodeLabel(child, nodeIndex, tree);
								} else {
									nodeList.add(nodeIndex);
								}

								this.indexToNode.put(nodeIndex, child);
							});
				else if (tree.isLeaf(treeIndex)) {
					if (labelsProvided)
						manageNodeLabel(parent, treeIndex, labelArr[labelToIndex.size()]);
					else
						manageNodeLabel(parent, treeIndex, tree);
				}
			}

		} else
			throw new Exception("Invalid input tree");
	}

	private void manageNodeLabel(TreeNode node, Integer nodeIndex, CompleteTree tree) {
		manageNodeLabel(node, nodeIndex, tree.getLabelIndex(nodeIndex));
	}

	private void manageNodeLabel(TreeNode node, Integer nodeIndex, int label) {
		node.label = label;
		this.labelToIndex.put(label, nodeIndex);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();

		sb.append("Degree: ");
		sb.append(k);
		sb.append(", ");

		sb.append("Labels/leaves: ");
		sb.append(m);
		sb.append(", ");

		sb.append("Number of internal nodes: ");
		sb.append(numberOfInternalNodes);
		sb.append(", ");

		sb.append("Size: ");
		sb.append(size);

		sb.append("\nTree:\n");
		sb.append(root.toString());

		return sb.toString();
	}

	@Override
	public int getRootIndex() {
		return root.index;
	}

	@Override
	public boolean hasLabel(int label) {
		return getAllLabels().contains(label);
	}
}
