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