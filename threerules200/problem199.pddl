(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(on-table b1)
		(clear b1)
		(on-table b2)
		(clear b2)
		(on-table b3)
		(clear b3)
		(on-table b4)
		(clear b4)
		(on-table b5)
		(clear b5)
		(on-table b6)
		(clear b6)
		(on-table b7)
		(clear b7)
		(on-table b8)
		(clear b8)
		(on-table b9)
		(clear b9)
		(in-tower t0)
		(clear t0)
		(midnightblue b0)
		(blue b0)
		(bisque b1)
		(orange b1)
		(darkorange b2)
		(orange b2)
		(hotpink b3)
		(pink b3)
		(lawngreen b4)
		(green b4)
		(lightgoldenrodyellow b5)
		(yellow b5)
		(darkviolet b6)
		(purple b6)
		(bisque b7)
		(orange b7)
		(darkred b8)
		(red b8)
		(deeppink b9)
		(pink b9)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?y) (or (not (yellow ?y)) (exists (?x) (and (green ?x) (on ?x ?y))))) (forall (?x) (or (not (purple ?x)) (exists (?y) (and (orange ?y) (on ?x ?y)))))))
)