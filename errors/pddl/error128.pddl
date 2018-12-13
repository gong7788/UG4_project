(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(clear b0)
		(on-table b1)
		(clear b1)
		(on-table b2)
		(clear b2)
		(on-table b3)
		(clear b3)
		(on-table b5)
		(clear b5)
		(on-table b9)
		(clear b9)
		(in-tower t0)
		(on b8 t0)
		(in-tower b8)
		(on b6 b8)
		(in-tower b6)
		(on b4 b6)
		(in-tower b4)
		(on b7 b4)
		(in-tower b7)
		(on b0 b7)
		(in-tower b0)
		(blue b0)
		(red b3)
		(green b4)
		(yellow b4)
		(green b6)
		(yellow b6)
		(green b7)
		(green b8)
		(yellow b8)
		(green b9)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?y) (or (not (yellow ?y)) (exists (?x) (and (green ?x) (on ?x ?y))))) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (not (on b3 b0))))
)