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
		(limegreen b0)
		(green b0)
		(limegreen b1)
		(green b1)
		(blueviolet b2)
		(purple b2)
		(orange b3)
		(bisque b4)
		(orange b4)
		(palegreen b5)
		(green b5)
		(forestgreen b6)
		(green b6)
		(fuchsia b7)
		(pink b7)
		(lightpink b8)
		(pink b8)
		(royalblue b9)
		(blue b9)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y)))))))
)