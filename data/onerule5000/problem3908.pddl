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
		(firebrick b0)
		(red b0)
		(yellowgreen b1)
		(green b1)
		(lightgoldenrodyellow b2)
		(yellow b2)
		(darkorchid b3)
		(purple b3)
		(dodgerblue b4)
		(blue b4)
		(darkblue b5)
		(blue b5)
		(lawngreen b6)
		(green b6)
		(red b7)
		(purple b8)
		(blueviolet b9)
		(purple b9)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y)))))))
)