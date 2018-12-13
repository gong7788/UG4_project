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
		(forestgreen b1)
		(green b1)
		(navy b2)
		(blue b2)
		(darkolivegreen b3)
		(green b3)
		(firebrick b4)
		(red b4)
		(darkorchid b5)
		(purple b5)
		(blue b6)
		(darkviolet b7)
		(purple b7)
		(darkorchid b8)
		(purple b8)
		(lawngreen b9)
		(green b9)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?x) (or (not (pink ?x)) (exists (?y) (and (red ?y) (on ?x ?y))))) (forall (?y) (or (not (red ?y)) (exists (?x) (and (purple ?x) (on ?x ?y)))))))
)