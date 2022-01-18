package main

func main() {
	a := &aa{}
	a.bbf = "abc"
}

type aa struct {
	cc string
	bb
}

func (b *aa) b() int {
	return 1
}

type bb struct {
	bbf string
}

func (b *bb) b() int {
	return 1
}
