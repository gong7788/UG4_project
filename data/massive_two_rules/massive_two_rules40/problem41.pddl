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
		(yellow b1)
		(palegreen b2)
		(green b2)
		(firebrick b3)
		(red b3)
		(maroon b4)
		(red b4)
		(darkorchid b5)
		(purple b5)
		(bisque b6)
		(orange b6)
		(darkmagenta b7)
		(purple b7)
		(bisque b8)
		(orange b8)
		(orange b9)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (green ?x)) (exists (?y) (and (orange ?y) (on ?x ?y))))) (forall (?x) (or (not (yellow ?x)) (exists (?y) (and (red ?y) (on ?x ?y)))))))
)