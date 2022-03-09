package algo

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCreateBT(t *testing.T) {

	values := []int{60, 40, 80, 35, 0, 78, 91}
	root := CreateBT(values)

	assert.Equal(t, 60, root.Value)
	assert.Equal(t, 35, root.Left.Left.Value)
	assert.Equal(t, true, root.Left.Right == nil)

	assert.Equal(t, 91, root.Right.Right.Value)
}

func TestPreOrderByRecursive(t *testing.T) {

	values := []int{60, 40, 80, 35, 0, 78, 91}
	root := CreateBT(values)
	_, res := root.PreOrderByRecursive()
	assert.Equal(t, []int{60, 40, 35, 80, 78, 91}, res, "TestPreOrderByRecursive---success")
	t.Logf("res : %v", res)
}
func TestPreOrder(t *testing.T) {

	values := []int{60, 40, 80, 35, 50, 0, 91, 0, 0, 45}
	root := CreateBT(values)
	_, res := root.PreOrderByRecursive()
	_, res2 := root.PreOrder()
	assert.Equal(t, res, res2, "TestPreOrderByRecursive---success")
	t.Logf("res : %v", res2)
}

func TestInOrderByRecursive(t *testing.T) {

	values := []int{60, 40, 80, 35, 0, 78, 91}
	root := CreateBT(values)
	_, res := root.InOrderByRecursive()
	assert.Equal(t, []int{35, 40, 60, 78, 80, 91}, res, "TestInOrderByRecursive---success")
	t.Logf("res : %v", res)
}

func TestInOrder(t *testing.T) {

	values := []int{60, 40, 80, 35, 50, 0, 91, 0, 0, 45}
	root := CreateBT(values)
	_, res := root.InOrderByRecursive()
	_, res2 := root.InOrder()
	assert.Equal(t, res, res2, "TestInOrder---success")
	t.Logf("res : %v", res)
}

func TestPostOrderByRecursive(t *testing.T) {

	values := []int{60, 40, 80, 35, 0, 78, 91}
	root := CreateBT(values)
	_, res := root.PostOrderByRecursive()
	assert.Equal(t, []int{35, 40, 78, 91, 80, 60}, res, "TestPostOrderByRecursive---success")
	t.Logf("res : %v", res)
}
func TestPostOrder(t *testing.T) {

	values := []int{60, 40, 80, 35, 50, 0, 91, 0, 0, 45}
	root := CreateBT(values)
	_, res := root.PostOrderByRecursive()
	_, res2 := root.PostOrder()
	assert.Equal(t, res, res2, "TestPostOrder---success")
	t.Logf("res : %v", res)
}
