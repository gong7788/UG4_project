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
		(indigo b0)
		(purple b0)
		(dodgerblue b1)
		(blue b1)
		(deepskyblue b2)
		(blue b2)
		(yellow b3)
		(lightyellow b4)
		(yellow b4)
		(maroon b5)
		(red b5)
		(darkorange b6)
		(orange b6)
		(darkorchid b7)
		(purple b7)
		(bisque b8)
		(orange b8)
		(palegreen b9)
		(green b9)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y)))))))
)