(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(on-table b5)
		(clear b5)
		(clear b7)
		(in-tower t0)
		(on b9 t0)
		(in-tower b9)
		(on b8 b9)
		(in-tower b8)
		(on b6 b8)
		(in-tower b6)
		(on b4 b6)
		(in-tower b4)
		(on b3 b4)
		(in-tower b3)
		(on b2 b3)
		(in-tower b2)
		(on b1 b2)
		(in-tower b1)
		(on b7 b1)
		(in-tower b7)
		(red b5)
		(red b7)
		(blue b5)
		(blue b0)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?y) (or (not (yellow ?y)) (exists (?x) (and (green ?x) (on ?x ?y))))) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (and (not (on b7 b8)) (not (on b7 b6)) (not (on b5 b6)) (not (on b7 b4)) (not (on b5 b4)) (not (on b7 b3)) (not (on b5 b3)) (not (on b7 b2)) (not (on b5 b2)) (not (on b5 b7)))))
)