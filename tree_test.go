package algo

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestFind(t *testing.T)  {

	root := &Node23{Values: []int{30,60}, Total: 2}

	level21 := &Node23{Values: []int{10,20}, Total: 2,Parent: root}
	level22 := &Node23{Values: []int{40}, Total: 1,Parent: root}
	level23 := &Node23{Values: []int{80}, Total: 1,Parent: root}
	root.Children = []*Node23{level21,level22,level23}

	level31 := &Node23{Values: []int{8}, Total: 1,Parent: level21}
	level32 := &Node23{Values: []int{15,16}, Total: 2,Parent: level21}
	level33 := &Node23{Values: []int{25}, Total: 1,Parent: level21}
	level21.Children = []*Node23{level31,level32,level33}

	level34 := &Node23{Values: []int{35}, Total: 1,Parent: level22}
	level35 := &Node23{Values: []int{50}, Total: 1,Parent: level22}
	level22.Children = []*Node23{level34,level35}

	level36 := &Node23{Values: []int{78}, Total: 1,Parent: level23}
	level37 := &Node23{Values: []int{91}, Total: 1,Parent: level23}
	level23.Children = []*Node23{level36,level37}

	t.Logf("root %v", root)

	//assert.Equal(t, true, true)
	//has
	hasVal, cur,parent := root.Find(30)
	t.Logf("cur %v ||| parent %v", cur, parent)
	assert.Equal(t, true, hasVal)

	hasVal, cur,parent = root.Find(8)
	t.Logf("cur %v ||| parent %v", cur, parent)
	assert.Equal(t, true, hasVal)

	hasVal, cur,parent = root.Find(15)
	t.Logf("cur %v ||| parent %v", cur, parent)
	assert.Equal(t, true, hasVal)

	hasVal, cur,parent = root.Find(78)
	t.Logf("cur %v ||| parent %v", cur, parent)
	assert.Equal(t, true, hasVal)
	assert.Equal(t, 2, len(root.Children[2].Children), "80 has two children")

	// exist
	hasVal, _, _ = root.Find(1)
	assert.Equal(t, false, hasVal)

	hasVal, _, _ = root.Find(100)
	assert.Equal(t, false, hasVal)

	hasVal, _, _ = root.Find(90)
	assert.Equal(t, false, hasVal)

	hasVal, _, _ = root.Find(17)
	assert.Equal(t, false, hasVal)
}

func TestInsertNode23(t *testing.T) {

	//待插入的元素已存在
	root := &Node23{Values: []int{30}, Total: 1}
	assert.Equal(t, root, root.Insert(30))

	//插入元素到空树
	root = &Node23{Total: 0}
	assert.Equal(t, root, root.Insert(30))
	assert.Equal(t, 30, root.Values[0])

	//插入根为2的树
	//root := &Node23{Values: []int{30}, Total: 1}
	assert.Equal(t, root, root.Insert(60))
	assert.Equal(t, 30, root.Values[0])

	//插入根节点为3的树
	root2 := &Node23{Values: []int{30}, Total: 1}
	root2.Insert(50)
	root2.Insert(10)
	assert.Equal(t, 30, root2.Values[0])
	assert.Equal(t, 10, root2.Children[0].Values[0])
	assert.Equal(t, 50, root2.Children[1].Values[0])

	//插入叶子节点为2的树
	root3 := &Node23{Values: []int{10}, Total: 1}
	root3.Insert(9)
	root3.Insert(11)
	root3.Insert(8)
	assert.Equal(t, 10, root3.Values[0])
	assert.Equal(t, 11, root3.Children[1].Values[0])
	assert.Equal(t, 8, root3.Children[0].Values[0])
	assert.Equal(t, 9, root3.Children[0].Values[1])

	//父节点为2待插入的为3
	root4 := &Node23{Values: []int{10}, Total: 1}
	root4.Insert(9)
	root4.Insert(11)
	root4.Insert(8)
	root4.Insert(7)
	assert.Equal(t, 8, root4.Values[0])
	assert.Equal(t, 10, root4.Values[1])
	assert.Equal(t, 7, root4.Children[0].Values[0])
	assert.Equal(t, 9, root4.Children[1].Values[0])
	assert.Equal(t, 11, root4.Children[2].Values[0])

	//父节点为3待插入的为3 且路径上有一个2节点
	root5 := &Node23{Values: []int{30}, Total: 1}

	level21 := &Node23{Total: 2, Values: []int{10,20}, Parent: root5}
	level22 := &Node23{Total: 1, Values: []int{80}, Parent: root5}
	root5.Children = []*Node23{level21,level22}

	level31 := &Node23{Total: 1, Values: []int{8}, Parent: level21}
	level32 := &Node23{Total: 2, Values: []int{15,16}, Parent: level21}
	level33 := &Node23{Total: 1, Values: []int{25}, Parent: level21}
	level21.Children = []*Node23{level31, level32, level33}
	level34 := &Node23{Total: 1, Values: []int{50}, Parent: level22}
	level35 := &Node23{Total: 1, Values: []int{90}, Parent: level22}
	level22.Children = []*Node23{level34, level35}


	root5.Insert(14)
	assert.Equal(t, []int{15,30}, root5.Values)
	assert.Equal(t, []int{10}, root5.Children[0].Values)
	assert.Equal(t, []int{20}, root5.Children[1].Values)
	assert.Equal(t, []int{80}, root5.Children[2].Values)

	//父节点为3待插入的为3 且路径上都是3节点
	root6 := &Node23{Values: []int{30,60}, Total: 1}
	level621 := &Node23{Total: 2, Values: []int{10,20}, Parent: root6}
	level622 := &Node23{Total: 1, Values: []int{40}, Parent: root6}
	level623 := &Node23{Total: 1, Values: []int{80}, Parent: root6}
	root6.Children = []*Node23{level621,level622, level623}

	level631 := &Node23{Total: 1, Values: []int{8}, Parent: level621}
	level632 := &Node23{Total: 2, Values: []int{15,16}, Parent: level621}
	level633 := &Node23{Total: 1, Values: []int{25}, Parent: level621}
	level621.Children = []*Node23{level631, level632, level633}

	level634 := &Node23{Total: 1, Values: []int{35}, Parent: level622}
	level635 := &Node23{Total: 1, Values: []int{50}, Parent: level622}
	level622.Children = []*Node23{level634, level635}

	level636 := &Node23{Total: 1, Values: []int{78}, Parent: level623}
	level637 := &Node23{Total: 1, Values: []int{91}, Parent: level623}
	level623.Children = []*Node23{level636, level637}

	newRoot6 := root6.Insert(17)
	assert.NotEqual(t, root6, newRoot6)
	assert.Equal(t, []int{30}, newRoot6.Values)

	assert.Equal(t, []int{16}, newRoot6.Children[0].Values)
	assert.Equal(t, []int{60}, newRoot6.Children[1].Values)

	assert.Equal(t, []int{10}, newRoot6.Children[0].Children[0].Values)
	assert.Equal(t, []int{20}, newRoot6.Children[0].Children[1].Values)
	assert.Equal(t, []int{40}, newRoot6.Children[1].Children[0].Values)
	assert.Equal(t, []int{80}, newRoot6.Children[1].Children[1].Values)

	assert.Equal(t, []int{8}, newRoot6.Children[0].Children[0].Children[0].Values)
	assert.Equal(t, []int{15}, newRoot6.Children[0].Children[0].Children[1].Values)
	assert.Equal(t, []int{17}, newRoot6.Children[0].Children[1].Children[0].Values)
	assert.Equal(t, []int{25}, newRoot6.Children[0].Children[1].Children[1].Values)

	assert.Equal(t, []int{35}, newRoot6.Children[1].Children[0].Children[0].Values)
	assert.Equal(t, []int{50}, newRoot6.Children[1].Children[0].Children[1].Values)
	assert.Equal(t, []int{78}, newRoot6.Children[1].Children[1].Children[0].Values)
	assert.Equal(t, []int{91}, newRoot6.Children[1].Children[1].Children[1].Values)
}

func TestInOrder(t *testing.T) {

	root := &Node23{Values: []int{30,60}, Total: 2}

	level21 := &Node23{Total: 2, Values: []int{10,20}, Parent: root}
	level22 := &Node23{Total: 1, Values: []int{40}, Parent: root}
	level23 := &Node23{Total: 1, Values: []int{80}, Parent: root}
	root.Children = []*Node23{level21,level22, level23}

	level31 := &Node23{Total: 1, Values: []int{8}, Parent: level21}
	level32 := &Node23{Total: 2, Values: []int{15,16}, Parent: level21}
	level33 := &Node23{Total: 1, Values: []int{25}, Parent: level21}
	level21.Children = []*Node23{level31,level32, level33}

	level34 := &Node23{Total: 1, Values: []int{35}, Parent: level22}
	level35 := &Node23{Total: 1, Values: []int{50}, Parent: level22}
	level22.Children = []*Node23{level34,level35}

	level36 := &Node23{Total: 1, Values: []int{78}, Parent: level23}
	level37 := &Node23{Total: 1, Values: []int{91}, Parent: level23}
	level23.Children = []*Node23{level36,level37}

	in := root.inOrder()
	t.Log(in)
}