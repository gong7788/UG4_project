(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(clear b1)
		(on-table b2)
		(clear b2)
		(on-table b4)
		(clear b4)
		(in-tower t0)
		(on b3 t0)
		(in-tower b3)
		(on b0 b3)
		(in-tower b0)
		(on b9 b0)
		(in-tower b9)
		(on b8 b9)
		(in-tower b8)
		(on b6 b8)
		(in-tower b6)
		(on b7 b6)
		(in-tower b7)
		(on b5 b7)
		(in-tower b5)
		(on b1 b5)
		(in-tower b1)
		(green b1)
		(yellow b1)
		(green b2)
		(blue b3)
		(green b4)
		(yellow b5)
		(red b6)
		(yellow b7)
		(blue b8)
		(red b9)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?y) (or (not (yellow ?y)) (exists (?x) (and (green ?x) (on ?x ?y))))) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (and (not (on b9 b3)) (not (on b6 b3)) (not (on b4 b5)) (not (on b2 b1)))))
)