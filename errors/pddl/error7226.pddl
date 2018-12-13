(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b1)
		(clear b1)
		(clear b6)
		(on-table b7)
		(clear b7)
		(in-tower t0)
		(on b9 t0)
		(in-tower b9)
		(on b8 b9)
		(in-tower b8)
		(on b5 b8)
		(in-tower b5)
		(on b4 b5)
		(in-tower b4)
		(on b0 b4)
		(in-tower b0)
		(on b3 b0)
		(in-tower b3)
		(on b2 b3)
		(in-tower b2)
		(on b6 b2)
		(in-tower b6)
		(green b1)
		(yellow b1)
		(red b6)
		(blue b6)
		(green b7)
		(yellow b7)
		(green b6)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?y) (or (not (yellow ?y)) (exists (?x) (and (green ?x) (on ?x ?y))))) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (and (not (on b7 b8)) (not (on b6 b8)) (not (on b7 b5)) (not (on b6 b5)) (not (on b7 b4)) (not (on b6 b0)) (not (on b7 b0)) (not (on b7 b3)) (not (on b6 b3)) (not (on b7 b2)) (not (on b7 b6)))))
)