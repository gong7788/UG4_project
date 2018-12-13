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
		(deepskyblue b0)
		(blue b0)
		(mediumblue b1)
		(blue b1)
		(blueviolet b2)
		(purple b2)
		(blueviolet b3)
		(purple b3)
		(cornflowerblue b4)
		(blue b4)
		(lightyellow b5)
		(yellow b5)
		(bisque b6)
		(orange b6)
		(lightyellow b7)
		(yellow b7)
		(magenta b8)
		(pink b8)
		(lightgoldenrodyellow b9)
		(yellow b9)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y)))))))
)