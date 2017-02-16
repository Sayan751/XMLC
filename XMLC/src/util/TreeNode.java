package util;

import java.util.ArrayList;
import java.util.List;

public class TreeNode {
	public int index;
	private int depth;
	public int label;
	public TreeNode parent;
	public List<TreeNode> children;

	/**
	 * @return the parent
	 */
	public TreeNode getParent() {
		return parent;
	}

	/**
	 * @param parent
	 *            the parent to set
	 */
	public void setParent(TreeNode parent) {
		if (this.parent != null)
			this.parent.children.remove(this);

		this.parent = parent;

		if (this.parent != null) {
			this.parent.children.add(this);
			depth = this.parent.depth + 1;
		} else {
			depth = 1;
		}
	}

	/**
	 * @return the depth
	 */
	public int getDepth() {
		return depth;
	}

	public TreeNode() {
	}

	public TreeNode(int index) {
		this(index, null, -1);
	}

	public TreeNode(int index, TreeNode parent) {
		this(index, parent, -1);
	}

	public TreeNode(int index, TreeNode parent, int label) {
		this.index = index;
		this.label = label;
		setParent(parent);
		this.children = new ArrayList<TreeNode>();
	}

	public boolean isLeaf() {
		if (this.children.size() > 0)
			return false;
		return true;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		buildString(1, sb);
		return sb.toString();
	}

	private void buildString(int indent, StringBuilder sb) {
		sb.append(String.format("%" + indent + "s", " "));
		if (isLeaf())
			sb.append("|-(" + index + "," + label + ")" + " (depth: " + depth + ")\n");
		else {
			sb.append("|-" + index + " (depth: " + depth + ")\n");
			children.stream()
					.forEach(child -> child.buildString(indent + 1, sb));
		}
	}
}