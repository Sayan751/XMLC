package util;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;

public class AdaptiveTreeTest {

	AdaptiveTree T;

	@Test
	public void Before_Adapting_Properties_Matches_to_Original_Tree() {

		try {
			// arrange
			CompleteTree tree = new CompleteTree(2, 7);

			// act
			T = new AdaptiveTree(tree, CompleteTree.name);
			// System.out.println(T);

			// assert
			assertEquals(tree.k, T.k);
			assertEquals(tree.m, T.m);
			assertEquals(tree.numberOfInternalNodes, T.numberOfInternalNodes);
			assertEquals(tree.size, T.size);
			assertEquals(4, T.getTreeDepth());
		} catch (Exception e) {
			e.printStackTrace();
			fail();
		}

	}

	@Test
	public void After_Adapting_Properties_Changes_Corretly() {

		try {
			// arrange
			CompleteTree tree = new CompleteTree(2, 7);
			T = new AdaptiveTree(tree, CompleteTree.name);
			int depth = T.getTreeDepth();

			// act
			T.adaptLeaf(0, 700);
			// System.out.println(T);

			// assert
			assertEquals(tree.k, T.k);
			assertEquals(tree.m + 1, T.m);
			assertEquals(tree.numberOfInternalNodes + 1, T.numberOfInternalNodes);
			assertEquals(tree.size + 2, T.size);
			assertEquals(depth, T.getTreeDepth());

		} catch (Exception e) {
			e.printStackTrace();
			fail();
		}
	}

	@Test
	public void Adapting_Leaf_Of_Non_Full_Parent_Adds_The_New_Leaf_To_Parent() {

		try {
			// arrange
			CompleteTree tree = new CompleteTree(3, 6);
			T = new AdaptiveTree(tree, CompleteTree.name);
			int newLabelIndex = 700;
			int depth = T.getTreeDepth();

			// act
			T.adaptLeaf(5, newLabelIndex);
			// System.out.println(T);

			// assert
			assertEquals(tree.k, T.k);
			assertEquals(tree.m + 1, T.m);
			assertEquals(tree.size + 1, T.size);
			assertTrue(T.getChildNodes(2)
					.contains(T.getTreeIndex(newLabelIndex)));
			assertEquals(depth, T.getTreeDepth());

		} catch (Exception e) {
			e.printStackTrace();
			fail();
		}
	}

	@Test
	public void Adapting_Full_Tree_Changes_Depth() {

		try {
			// arrange
			CompleteTree tree = new CompleteTree(3, 9);
			T = new AdaptiveTree(tree, CompleteTree.name);
			int newLabelIndex = 700;
			int depth = T.getTreeDepth();

			// act
			T.adaptLeaf(8, newLabelIndex);
			// System.out.println(T);

			// assert
			assertEquals(tree.k, T.k);
			assertEquals(tree.m + 1, T.m);
			assertEquals(tree.size + 2, T.size);
			assertTrue(T.getChildNodes(3)
					.contains(14));
			assertTrue(T.getChildNodes(14)
					.contains(12));
			assertTrue(T.getChildNodes(14)
					.contains(13));
			assertEquals(depth + 1, T.getTreeDepth());

		} catch (Exception e) {
			e.printStackTrace();
			fail();
		}
	}

}
