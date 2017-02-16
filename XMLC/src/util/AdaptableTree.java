package util;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class AdaptableTree extends Tree {
	TreeNode tree;
	Map<Integer, TreeNode> indexToNode = new HashMap<Integer, TreeNode>();
	Map<Integer, Integer> labelToIndex = new HashMap<Integer, Integer>();

	public AdaptableTree() {
	}

	public AdaptableTree(Tree tree, String treeType) throws Exception {
		// Get the basic details
		this.m = tree.m;
		this.k = tree.k;
		this.size = tree.size;
		this.numberOfInternalNodes = tree.numberOfInternalNodes;

		switch (treeType) {
		case CompleteTree.name:
			populateFromCompleteTree((CompleteTree) tree);
			break;
		default:
			throw new Exception(String.format("AdaptableTree for %s is not yet implemented.", treeType));
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
		return this.labelToIndex.get(label);
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
		return computeTreeDepth(tree);
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

		if (parent.children.size() < k) {
			newLeaf.setParent(parent);
		} else {
			maxTreeIndex++;
			TreeNode newParent = new TreeNode(maxTreeIndex, parent);
			indexToNode.put(maxTreeIndex, newParent);

			leaf.setParent(newParent);
			newLeaf.setParent(newParent);

			size++;
			numberOfInternalNodes++;
		}
		return newLeaf.index;
	}

	private void populateFromCompleteTree(CompleteTree tree) throws Exception {
		if (tree != null) {
			// Start with root
			int treeIndex = 0;
			this.tree = new TreeNode(treeIndex);
			indexToNode.put(treeIndex, this.tree);

			List<Integer> nodeList = new ArrayList<Integer>();
			nodeList.add(treeIndex);

			while (!nodeList.isEmpty()) {
				treeIndex = nodeList.remove(0);
				TreeNode parent = indexToNode.get(treeIndex);

				tree.getChildNodes(treeIndex)
						.stream()
						.forEach(nodeIndex -> {

							TreeNode child = new TreeNode(nodeIndex, parent);

							if (tree.isLeaf(nodeIndex)) {
								int label = tree.getLabelIndex(nodeIndex);
								child.label = label;
								this.labelToIndex.put(label, nodeIndex);
							} else {
								nodeList.add(nodeIndex);
							}

							this.indexToNode.put(nodeIndex, child);
						});
			}

		} else
			throw new Exception("Invalid input tree");
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
		sb.append(tree.toString());

		return sb.toString();
	}

	public static void main(String[] args) throws Exception {
		AdaptableTree T = new AdaptableTree(new CompleteTree(2, 7), CompleteTree.name);
		System.out.println(T);

		T.adaptLeaf(0, 700);
		System.out.println(T);
		
		// System.out.println("---------------");
		// T.adaptLeaf(0, 700);
		// T.printToConsole();
		//
		// System.out.println("---------------");
		// T = new AdaptableTree(new CompleteTree(3, 6), CompleteTree.name);
		// T.printToConsole();
		// System.out.println("---------------");
		// T.adaptLeaf(0, 700);
		// T.printToConsole();
		//
		// System.out.println("---------------");
		// T = new AdaptableTree(new CompleteTree(2, 8), CompleteTree.name);
		// T.printToConsole();
		// System.out.println("---------------");
		// T.adaptLeaf(7, 700);
		// T.printToConsole();
	}
}
