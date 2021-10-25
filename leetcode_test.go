package algo

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBinarySearch(t *testing.T) {

	testSlice := []int{0, 1, 3, 4, 5, 6}
	res := BinarySearch(testSlice, 2)
	assert.Equal(t, false, res)

	testSlice = []int{0, 1, 3, 4, 5, 6}
	res = BinarySearch(testSlice, 0)
	assert.Equal(t, true, res)

	testSlice = []int{0, 1, 3, 4, 5, 6}
	res = BinarySearch(testSlice, 6)
	assert.Equal(t, true, res)

	testSlice = []int{0, 1, 3, 4, 5, 6}
	res = BinarySearch(testSlice, -1)
	assert.Equal(t, false, res)

	testSlice = []int{0, 1, 3, 4, 5, 6}
	res = BinarySearch(testSlice, 7)
	assert.Equal(t, false, res)

	testSlice = []int{0, 1, 3, 4, 5, 6}
	res = BinarySearch(testSlice, 5)
	assert.Equal(t, true, res)
}

func TestRemoveElement(t *testing.T) {

	testSlice := []int{1, 1, 2, 3, 1, 3}
	res := removeElement(testSlice, 1)
	assert.Equal(t, 3, res)
	res = removeElement(testSlice, 3)
	assert.Equal(t, 4, res)
	res = removeElement(testSlice, 2)
	assert.Equal(t, 5, res)

	testSlice = []int{1}
	res = removeElement(testSlice, 1)
	assert.Equal(t, 0, res)
	res = removeElement(testSlice, 2)
	assert.Equal(t, 1, res)
}

func TestRemoveElement2(t *testing.T) {

	testSlice := []int{1, 1, 2, 3, 1, 3}
	res := removeElement2(testSlice, 1)
	assert.Equal(t, 3, res)

	testSlice = []int{1, 1, 2, 3, 1, 3}
	res = removeElement2(testSlice, 3)
	assert.Equal(t, 4, res)

	testSlice = []int{1, 1, 2, 3, 1, 3}
	res = removeElement2(testSlice, 2)
	assert.Equal(t, 5, res)
}

func TestSortedSquares(t *testing.T) {

	testSlice := []int{-4, -1, 0, 3, 10}
	res := sortedSquares(testSlice)
	assert.Equal(t, []int{0, 1, 9, 16, 100}, res)
}

func TestSortedSquares2(t *testing.T) {

	testSlice := []int{-4, -1, 0, 3, 10}
	res := sortedSquares2(testSlice)
	assert.Equal(t, []int{0, 1, 9, 16, 100}, res)
}

func TestMinSubArrayLen(t *testing.T) {

	testSlice := []int{2, 3, 1, 2, 4, 3}
	res := minSubArrayLen(7, testSlice)
	assert.Equal(t, 2, res)

	res = minSubArrayLen(100, testSlice)
	assert.Equal(t, 0, res)

	res = minSubArrayLen(1, testSlice)
	assert.Equal(t, 1, res)
}

func TestMinSubArrayLen2(t *testing.T) {

	testSlice := []int{2, 3, 1, 2, 4, 3}
	res := minSubArrayLen2(7, testSlice)
	assert.Equal(t, 2, res)

	res = minSubArrayLen2(100, testSlice)
	assert.Equal(t, 0, res)

	res = minSubArrayLen2(1, testSlice)
	assert.Equal(t, 1, res)
}

func TestGenerateMatrix(t *testing.T) {

	res := generateMatrix(5)

	assert.EqualValues(t, [][]int{{1, 2, 3, 4, 5}, {16, 17, 18, 19, 6}, {15, 24, 25, 20, 7}, {14, 23, 22, 21, 8}, {13, 12, 11, 10, 9}}, res)
}

func TestRemoveElements(t *testing.T) {

	testSlice := []int{1, 2, 6, 3, 4, 5, 6}
	head := GeneralLinkList(testSlice)
	res := removeElements(head, 1)
	assert.Equal(t, 2, res.Val)
	assert.Equal(t, 6, res.Next.Val)

	testSlice = []int{1, 2, 6, 3, 4, 5, 6}
	head = GeneralLinkList(testSlice)
	res = removeElements(head, 2)
	assert.Equal(t, 1, res.Val)
	assert.Equal(t, 6, res.Next.Val)

}

func TestReverseList(t *testing.T) {

	testSlice := []int{1, 2, 3, 4}

	res := GeneralLinkList(testSlice)
	newHead := reverseList(res)
	assert.Equal(t, 4, newHead.Val)
	assert.Equal(t, 3, newHead.Next.Val)
	assert.Equal(t, 2, newHead.Next.Next.Val)
	assert.Equal(t, 1, newHead.Next.Next.Next.Val)
}

func TestSwapPairs(t *testing.T) {

	testSlice := []int{1, 2, 3, 4}
	res := GeneralLinkList(testSlice)
	swap := swapPairs(res)
	t.Logf("swap %v", swap)
	swapSlice := swap.getListValues()
	assert.Equal(t, []int{2, 1, 4, 3}, swapSlice)

}

func TestRemoveNthFromEnd(t *testing.T) {

	testSlice := []int{1, 2, 3, 4, 5}
	head := GeneralLinkList(testSlice)
	after := removeNthFromEnd(head, 2)
	assert.Equal(t, []int{1, 2, 3, 5}, after.getListValues())

	testSlice = []int{1, 2, 3, 4, 5}
	head = GeneralLinkList(testSlice)
	after = removeNthFromEnd(head, 1)
	assert.Equal(t, []int{1, 2, 3, 4}, after.getListValues())

	testSlice = []int{1, 2, 3, 4, 5}
	head = GeneralLinkList(testSlice)
	after = removeNthFromEnd(head, 5)
	assert.Equal(t, []int{2, 3, 4, 5}, after.getListValues())

}

func TestRemoveNthFromEnd2(t *testing.T) {

	testSlice := []int{1, 2, 3, 4, 5}
	head := GeneralLinkList(testSlice)
	after := removeNthFromEnd2(head, 2)
	assert.Equal(t, []int{1, 2, 3, 5}, after.getListValues())

	testSlice = []int{1, 2, 3, 4, 5}
	head = GeneralLinkList(testSlice)
	after = removeNthFromEnd2(head, 1)
	assert.Equal(t, []int{1, 2, 3, 4}, after.getListValues())

	testSlice = []int{1, 2, 3, 4, 5}
	head = GeneralLinkList(testSlice)
	after = removeNthFromEnd2(head, 5)
	assert.Equal(t, []int{2, 3, 4, 5}, after.getListValues())

	testSlice = []int{1, 2, 3, 4, 5}
	head = GeneralLinkList(testSlice)
	after = removeNthFromEnd2(head, 6)
	assert.Equal(t, []int{1, 2, 3, 4, 5}, after.getListValues())

}

func TestGetIntersectionNode(t *testing.T) {

	slice1 := []int{1, 2, 3, 4, 5}
	head1 := GeneralLinkList(slice1)

	slice2 := []int{7, 8}
	head2 := GeneralLinkList(slice2)

	head2.Next.Next = head1.Next.Next //3
	assert.Equal(t, head2.Next.Next, GetIntersectionNode(head1, head2))
}

func TestDetectCycle(t *testing.T) {

	testSlice := []int{1, 2, 3, 4, 5}
	head := GeneralLinkList(testSlice)
	head.Next.Next.Next.Next.Next = head
	cycle := DetectCycle(head)
	assert.Equal(t, head, cycle)

}

func TestIsAnagram(t *testing.T) {

	res := IsAnagram("anagram", "nagaram")
	assert.Equal(t, true, res)

	res = IsAnagram("rat", "car")
	assert.Equal(t, false, res)
}

func TestIntersection(t *testing.T) {

	res := Intersection([]int{1, 2, 2, 1}, []int{2, 2})
	assert.Equal(t, []int{2}, res)

	res = Intersection([]int{4, 9, 5}, []int{9, 4, 9, 8, 4})
	assert.Equal(t, []int{9, 4}, res)

}

func TestIsHappy(t *testing.T) {

	// res := IsHappy(19)
	// assert.Equal(t, true, res)

	res := IsHappy(2)
	assert.Equal(t, false, res)
}

func TestTwoSum(t *testing.T) {

	testSlice := []int{2, 7, 11, 15}

	assert.Equal(t, []int{0, 1}, TwoSum(testSlice, 9))
}

func TestFourSumCount(t *testing.T) {

	assert.Equal(t, 2, FourSumCount([]int{1, 2}, []int{-2, -1}, []int{-1, 2}, []int{0, 2}))
}

func TestCanConstruct(t *testing.T) {

	assert.Equal(t, false, CanConstruct("a", "b"))
	assert.Equal(t, false, CanConstruct("aa", "ab"))
	assert.Equal(t, true, CanConstruct("aa", "aab"))
}

func TestThreeSum(t *testing.T) {
	_ = ThreeSum([]int{-1, 0, 1, 2, -1, -4})
}

func TestFourSum(t *testing.T) {

	_ = FourSum([]int{-2, -2, -2, -2, 1, 0, -1, 0, -2, 2}, 100)
}

func TestReverseString(t *testing.T) {

	test := "hello"
	ReverseString([]byte(test))
}

func TestReverseStr(t *testing.T) {

	test := "abcdefg"
	res := ReverseStr(test, 2)
	assert.Equal(t, "bacdfeg", res)

	test = "abcd"
	res = ReverseStr(test, 2)
	assert.Equal(t, "bacd", res)

}

func TestReplaceSpace(t *testing.T) {

	res := ReplaceSpace("We are happy")

	assert.Equal(t, "We%20are%20happy", res)
}

func TestReverseWords(t *testing.T) {
	res := ReverseWords("  Bob    Loves  Alice   ")
	assert.Equal(t, "Alice Loves Bob", res)
}

func TestReverseLeftWords(t *testing.T) {

	assert.Equal(t, "cdefgab", ReverseLeftWords("abcdefg", 2))
}

func TestRepeatedSubstringPattern(t *testing.T) {

	assert.Equal(t, true, RepeatedSubstringPattern("abab"))
	assert.Equal(t, false, RepeatedSubstringPattern("aba"))
	assert.Equal(t, true, RepeatedSubstringPattern("abcabcabcabc"))
}

func TestIsValid(t *testing.T) {

	assert.Equal(t, false, IsValid("}"))
	assert.Equal(t, true, IsValid("()"))
	assert.Equal(t, true, IsValid("()[]{}"))
	assert.Equal(t, false, IsValid("(]"))
	assert.Equal(t, false, IsValid("([)]"))
	assert.Equal(t, true, IsValid("{[]}"))
}

func TestRemoveDuplicates(t *testing.T) {

	assert.Equal(t, "ca", RemoveDuplicates("abbaca"))
}

func TestEvalRPN(t *testing.T) {

	assert.Equal(t, 22, EvalRPN([]string{"10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"}))
}

func TestMaxSlidingWindow(t *testing.T) {

	res := MaxSlidingWindow([]int{1, 3, -1, -3, 5, 3, 6, 7}, 3)

	assert.Equal(t, []int{3, 3, 5, 5, 6, 7}, res)

	res = MaxSlidingWindow([]int{1, 3, -1, -3, 5, 3, 6, 7}, 4)
	assert.Equal(t, []int{3, 5, 5, 6, 7}, res)

}

func TestInvertTree(t *testing.T) {

	testSlice := []int{4, 2, 7, 1, 3, 6, 9}

	root := GeneralTree(testSlice)

	res := InvertTree(root)
	level, _, _ := res.LevelOrder()
	assert.Equal(t, []int{4, 7, 2, 9, 6, 3, 1}, level)

}

func TestIsSymmetric(t *testing.T) {

	testSlice := []int{1, 2, 2, 3, 4, 4, 3}
	root := GeneralTree(testSlice)
	symmetric := IsSymmetric(root)

	assert.Equal(t, true, symmetric)

	testSlice = []int{1, 2, 2, -999, 3, -999, 3}
	root = GeneralTree(testSlice)
	symmetric = IsSymmetric(root)

	assert.Equal(t, false, symmetric)
}

func TestMaxDepth(t *testing.T) {

	testSlice := []int{3, 9, 20, -999, -999, 15, 7}
	root := GeneralTree(testSlice)

	assert.Equal(t, true, root.Left.Left == nil)
	assert.Equal(t, 3, MaxDepth(root))
}

func TestMinDepth(t *testing.T) {

	testSlice := []int{3, 9, 20, -999, -999, 15, 7}
	root := GeneralTree(testSlice)
	assert.Equal(t, 2, MinDepth(root))

	testSlice = []int{2, -999, 3, -999, 4, -999, 5, -999, 6}
	root = GeneralTree(testSlice)
	assert.Equal(t, 5, MinDepth(root))
}

func TestCountNodes(t *testing.T) {

	testSlice := []int{1, 2, 3, 4, 5, 6}
	root := GeneralTree(testSlice)
	assert.Equal(t, 6, CountNodes(root))

}

func TestIsBalance(t *testing.T) {

	testSlice := []int{1, 2, 2, 3, 3, 4, 4}
	root := GeneralTree(testSlice)
	assert.Equal(t, false, IsBalanced(root))

	testSlice = []int{3, 9, 20, -999, -999, 15, 17}
	root = GeneralTree(testSlice)
	assert.Equal(t, true, IsBalanced(root))

}

func TestBinaryTreePaths(t *testing.T) {

	testSlice := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	//testSlice := []int{1,2,-999,4,5,8,9,10,11}
	root := GeneralTree(testSlice)
	_ = BinaryTreePaths(root)

}

func TestSumOfLeftLeaves(t *testing.T) {

	testSlice := []int{3, 9, 20, -999, -999, 15, 7}
	//testSlice := []int{1,2,-999,4,5,8,9,10,11}
	root := GeneralTree(testSlice)
	res := SumOfLeftLeaves(root)
	assert.Equal(t, 24, res)
}

func TestFindBottomLeftValue(t *testing.T) {

	testSlice := []int{2, 1, 3}
	root := GeneralTree(testSlice)
	res := FindBottomLeftValue(root)
	assert.Equal(t, 1, res)

	testSlice = []int{1, 2, 3, 4, -999, 5, 6, -999, -999, 7}
	root = GeneralTree(testSlice)
	res = FindBottomLeftValue(root)
	assert.Equal(t, 7, res)

}

func TestHasPathSum(t *testing.T) {

	testSlice := []int{1, 2, 3}
	root := GeneralTree(testSlice)
	res := HasPathSum(root, 5)
	assert.Equal(t, false, res)

	testSlice = []int{5, 4, 8, 11, -999, 13, 4, 7, 2, -999, -999, -999, 1}
	root = GeneralTree(testSlice)
	l, _, _ := root.LevelOrder()
	t.Log(l)
	res = HasPathSum(root, 22)
	assert.Equal(t, true, res)
}

func TestBuildTree(t *testing.T) {

	root := BuildTree([]int{9, 3, 15, 20, 7}, []int{9, 15, 7, 20, 3})

	assert.Equal(t, 3, root.Val)
	assert.Equal(t, 9, root.Left.Val)
	assert.Equal(t, true, root.Left.Left == nil)

	assert.Equal(t, 20, root.Right.Val)
	assert.Equal(t, 7, root.Right.Right.Val)
	assert.Equal(t, true, root.Right.Right.Right == nil)

	assert.Equal(t, 15, root.Right.Left.Val)
	assert.Equal(t, true, root.Right.Left.Left == nil)

	prev, _ := root.PrevOrderByIteration()
	t.Log(prev)
}

func TestConstructMaximumBinaryTree(t *testing.T) {

	root := ConstructMaximumBinaryTree([]int{3, 2, 1, 6, 2, 5})

	assert.Equal(t, 6, root.Val)
	assert.Equal(t, 3, root.Left.Val)
	assert.Equal(t, true, root.Left.Left == nil)
	assert.Equal(t, 2, root.Left.Right.Val)
	assert.Equal(t, 1, root.Left.Right.Right.Val)

	t.Log(root.Right)
	//assert.Equal(t, 5, root.Right.Val)
	//todo
	//assert.Equal(t, 2, root.Right.Left.Val)

}

func TestMergeTrees(t *testing.T) {

	root1 := GeneralTree([]int{1, 3, 2, 5})
	root2 := GeneralTree([]int{2, 1, 3, -999, 4, -999, 7})

	root := MergeTrees(root1, root2)

	level, _, _ := root.LevelOrder()

	assert.Equal(t, []int{3, 4, 5, 5, 4, 7}, level)
}

func TestSearchBST(t *testing.T) {

	root := GeneralTree([]int{4, 2, 7, 1, 3})

	res := SearchBST(root, 5)
	assert.Equal(t, true, res == nil)

	res = SearchBST(root, 1)
	assert.Equal(t, 1, res.Val)

	res = SearchBST(root, 2)
	assert.Equal(t, 2, res.Val)
	assert.Equal(t, 1, res.Left.Val)
	assert.Equal(t, 3, res.Right.Val)

}

func TestIsValidBST(t *testing.T) {

	root := GeneralTree([]int{2, 1, 3})
	assert.Equal(t, true, IsValidBST(root))

	root = GeneralTree([]int{5, 1, 4, -999, -999, 3, 6})
	assert.Equal(t, false, IsValidBST(root))
}

func TestGetMinimumDifference(t *testing.T) {

	root := GeneralTree([]int{1, -999, 3, 2})

	assert.Equal(t, 1, GetMinimumDifference(root))
}

func TestFindMode(t *testing.T) {

	root := GeneralTree([]int{1, -999, 2, 2})

	assert.Equal(t, 2, FindMode(root))
}

func TestDeleteNode(t *testing.T) {

	root := GeneralTree([]int{5, 3, 6, 2, 4, -999, 7})

	le, _, _ := root.LevelOrder()
	assert.Equal(t, []int{5, 3, 6, 2, 4, 7}, le)

	pr, _ := root.InOrderByRecursive()
	assert.Equal(t, []int{2, 3, 4, 5, 6, 7}, pr)

	assert.Equal(t, 4, DeleteNode(root, 5).Val)

	root = GeneralTree([]int{5, 3, 6, 2, 4, -999, 7})
	assert.Equal(t, true, DeleteNode(root, 4).Left.Right == nil)

	root = GeneralTree([]int{5, 3, 6, 2, 4, -999, 7})
	assert.Equal(t, 7, DeleteNode(root, 6).Right.Val)

}

func TestTrimBst(t *testing.T) {

	root := GeneralTree([]int{6, 3, 8, 1, 5, 7, 9, 0, 2})
	newRoot := TrimBST(root, 3, 7)
	le, _, _ := newRoot.LevelOrder()
	assert.Equal(t, []int{6, 3, 7, 5}, le)

}

func TestConvertBST(t *testing.T) {

	root := GeneralTree([]int{4, 1, 6, 0, 2, 5, 7, -999, -999, -999, 3, -999, -999, -999, 8})
	newRoot := ConvertBST(root)
	le, _, _ := newRoot.LevelOrder()
	assert.Equal(t, []int{30, 36, 21, 36, 35, 26, 15, 33, 8}, le)
}

func TestCombine(t *testing.T) {

	_ = Combine(5, 3)
}

func TestCombinationSum3(t *testing.T) {

	_ = CombinationSum3(3, 9)
}

func TestLetterCombinations(t *testing.T) {
	_ = LetterCombinations("1234")
}

func TestPermute(t *testing.T) {

	_ = Permute([]int{1, 2, 3})
}

func TestPermute2(t *testing.T) {

	_ = Permute2([]int{1, 2, 3})
}

func TestCombinationSum(t *testing.T) {

	_ = CombinationSum([]int{2, 3, 5}, 8)
}

func TestCombinationSum2(t *testing.T) {

	_ = CombinationSum2([]int{10, 1, 2, 7, 6, 1, 5}, 8)
}

func TestPartition(t *testing.T) {

	_ = Partition("aabaa")
}

func TestRestoreIpAddresses(t *testing.T) {

	_ = RestoreIpAddresses("101023")
}

func TestSubsets(t *testing.T) {

	_ = Subsets([]int{1, 2, 3, 4})
}

func TestSubsetsWithDup(t *testing.T) {

	_ = SubsetsWithDup([]int{1, 2, 2})
}

func TestFindSubsequences(t *testing.T) {

    _ = FindSubsequences([]int{4, 6, 7, 7})
}


func TestPermuteUnique(t *testing.T) {

    _ = PermuteUnique([]int{1, 1, 2})
}

func TestSolveNQueens(t *testing.T) {
    _ = SolveNQueens(5)

}
